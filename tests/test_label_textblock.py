import numpy as np

from cartoweave.config.loader import load_data_defaults
from cartoweave.data.generate import generate_scene
from cartoweave.data.textblock import load_font, measure_text_block


def test_labels_have_text_and_wh():
    cfg = load_data_defaults("configs/data.yaml")
    assert cfg.generate is not None
    gen = cfg.generate
    rng = np.random.default_rng(0)
    P0, labels0, active0, scene0 = generate_scene(gen, rng)
    txt_cfg = gen.text
    font = load_font(txt_cfg.font.path, int(txt_cfg.font.size))
    len_min, len_max = map(int, txt_cfg.len_range)
    spacing = int(txt_cfg.line_spacing_px)
    padx = int(txt_cfg.padding_px.x)
    pady = int(txt_cfg.padding_px.y)
    for lbl in labels0:
        lines = lbl.meta.get("text_lines")
        assert lines is not None
        assert len(lines) in {1, 3}
        for s in lines:
            assert s.isalpha()
            assert len_min <= len(s) <= len_max
        W, H = measure_text_block(lines, font, spacing, padx, pady)
        assert lbl.WH == (float(W), float(H))
