.PHONY: help list verify test-toronto-single test-toronto-constellation test-shanghai-single test-shanghai-constellation test-all clean

help:
	@echo "可用的测试命令:"
	@echo "  make list                      - 列出所有可用场景"
	@echo "  make verify                    - 验证所有场景配置"
	@echo "  make test-toronto-single       - 运行Toronto单卫星场景"
	@echo "  make test-toronto-constellation - 运行Toronto星座场景"
	@echo "  make test-shanghai-single      - 运行Shanghai单卫星场景"
	@echo "  make test-shanghai-constellation - 运行Shanghai星座场景"
	@echo "  make test-all                  - 运行所有场景"
	@echo "  make clean                     - 清理Python缓存文件"
	@echo ""
	@echo "或使用通用命令:"
	@echo "  make test SCENARIO=toronto_single"

list:
	python run_test.py --list

verify:
	python verify_configs.py

test-toronto-single:
	python run_test.py --scenario toronto_single

test-toronto-constellation:
	python run_test.py --scenario toronto_constellation

test-shanghai-single:
	python run_test.py --scenario shanghai_single

test-shanghai-constellation:
	python run_test.py --scenario shanghai_constellation

test-all:
	python run_test.py --all

# 通用测试命令 (需要指定 SCENARIO 参数)
test:
ifdef SCENARIO
	python run_test.py --scenario $(SCENARIO)
else
	@echo "错误: 请指定场景名称"
	@echo "使用方法: make test SCENARIO=toronto_single"
	@echo "或使用: make test-toronto-single"
	@make list
endif

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "已清理Python缓存文件"
