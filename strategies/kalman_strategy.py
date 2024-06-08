from pykalman import KalmanFilter
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    Direction,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

class KalmanFilterStrategy(CtaTemplate):
    author = "Jack Du"

    # 策略参数
    trading_size = 1
    
    # 策略变量
    state_mean = 0.0
    state_cov = 1.0

    parameters = ["trading_size"]
    variables = ["state_mean", "state_cov"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            transition_covariance=0.01,
            observation_covariance=1
        )

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 获取最新的价格
        measurement = bar.close_price
        
        # 卡尔曼滤波预测和更新
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean,
            self.state_cov,
            observation=measurement
        )
        
        self.write_log(f"State Mean: {self.state_mean}, Measurement: {measurement}")
        
        # 基于卡尔曼滤波结果的交易逻辑
        if self.state_mean > measurement:
            self.buy(bar.close_price + 5, self.trading_size)
            self.write_log(f"Buy Order Sent: {bar.close_price + 5}")
        elif self.state_mean < measurement:
            self.short(bar.close_price - 5, self.trading_size)
            self.write_log(f"Short Order Sent: {bar.close_price - 5}")
        
        self.put_event()

    def on_order(self, order: OrderData):
        self.write_log(f"Order Status: {order.status}")

    def on_trade(self, trade: TradeData):
        self.write_log(f"Trade: {trade}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass
