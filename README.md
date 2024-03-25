## 智能调度算法
实验环境: torch -> 1.8.1+cu111

## 应用场景
1. 工序排产问题

## 使用方式
1. 测试基础调度器: `python tests/test_base_scheduler.py`
2. 启动viz可视化工具: `!python -m visdom.server`, 如果在服务器启动,需要进行端口映射: `ssh -L 18097:127.0.0.1:8097 root@[ip] -p [port]`, 服务器密码: [password], 接着在本地访问: `http://localhost:18097/`
3. 模型训练: `python train.py`
4. 模型测试: `python experiments.py [exp]` exp为1到14