import os
import torch
import torch.nn as nn
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

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        action_probs, state_value = self.actor(lstm_out), self.critic(lstm_out)
        return action_probs, state_value

class A2C_LSTM_Strategy(CtaTemplate):
    author = "Jack Du"
    
    # 策略参数
    trading_size = 1
    seq_length = 30  # 使用LSTM的时间步长
    
    # 策略变量
    state_mean = 0.0
    state_cov = 1.0

    parameters = ["trading_size"]
    variables = ["state_mean", "state_cov"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(self.seq_length + 1)  # 需要多一个bar来确保有足够的序列长度

        self.model_path = 'E:\课程\程序设计\期末\Investment-Strategy-with-RL\saved_models\ac_lstm_model.pth'
        self.input_dim = 5
        self.hidden_dim = 128
        self.action_dim = 3

        self.model = ActorCritic(self.input_dim, self.hidden_dim, self.action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 加载训练好的模型参数
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.write_log(f"Loaded model from {self.model_path}")

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

        if len(self.am.close_array) < self.seq_length:
            return

        # 获取状态的时间序列输入
        state = torch.tensor(self.am.close_array[-self.seq_length:], dtype=torch.float32)
        state = state.unsqueeze(0).unsqueeze(2)  # 调整形状为 (1, seq_length, 1)
        state = state.repeat(1, 1, self.input_dim)  # 调整形状为 (1, seq_length, input_dim)
        state = state.to(self.device)

        with torch.no_grad():
            action_probs, state_value = self.model(state)

        action = torch.argmax(action_probs).item()
        
        if action == 0:  # Buy
            self.buy(bar.close_price + 5, self.trading_size)
        elif action == 1:  # Sell
            self.sell(bar.close_price - 5, self.trading_size)
        elif action == 2:  # Hold
            pass
        
        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass
