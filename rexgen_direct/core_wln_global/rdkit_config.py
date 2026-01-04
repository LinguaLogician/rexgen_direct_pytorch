# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: rdkit_config.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 20:39
# https://chat.deepseek.com/a/chat/s/389b0c7a-682c-4239-b659-b881f53efe93
# rdkit_config.py
import warnings
from rdkit import RDLogger
import logging


def configure_rdkit_logging(level='ERROR'):
    """
    配置 RDKit 日志级别

    Args:
        level: 日志级别，可以是 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'OFF'
    """
    # 禁用所有 RDKit 日志
    if level == 'OFF':
        RDLogger.DisableLog('rdApp.*')
        return

    # 设置日志级别
    rdlogger = RDLogger.logger()

    if level == 'ERROR':
        rdlogger.setLevel(RDLogger.ERROR)
    elif level == 'WARNING':
        rdlogger.setLevel(RDLogger.WARNING)
    elif level == 'INFO':
        rdlogger.setLevel(RDLogger.INFO)
    elif level == 'DEBUG':
        rdlogger.setLevel(RDLogger.DEBUG)

    # 禁用弃用警告
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='rdkit')

    print(f"RDKit logging configured to level: {level}")


# 默认配置为 WARNING 级别（只显示错误和警告）
configure_rdkit_logging('WARNING')
