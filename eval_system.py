from eval import EvalRunner

runner = EvalRunner()
runner.load_dataset()
results = runner.run_all()
report = runner.generate_report()
runner.save_results()

# 查看历史趋势
storage = runner.storage
trend = storage.latest_trend(5)
print(trend)