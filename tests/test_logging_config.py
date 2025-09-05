
def test_logger_config_json_format(monkeypatch, capsys):
    from cartoweave.logging_util import get_logger
    cfg = {"compute": {"logging": {"level": "INFO", "format": "json"}}}
    logger = get_logger("cw.test", cfg)
    logger.info("hello", extra={"extra": {"k": 1}})
    out = capsys.readouterr().out.strip()
    assert out.startswith("{") and '"k": 1' in out and '"hello"' not in out
