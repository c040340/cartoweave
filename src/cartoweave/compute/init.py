from ..logging import init_logging

# 导入阶段先给一个保守等级（WARNING），避免噪音；运行时入口会再用 cfg 覆盖
init_logging("none")
