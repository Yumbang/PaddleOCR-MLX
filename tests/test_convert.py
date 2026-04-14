"""Tests for mlx_ppocr.convert (pure-logic functions)."""


from mlx_ppocr.convert import (
    _is_linear_weight,
    _map_bn,
    _map_conv_bn,
    _map_paddle_key,
    _map_rep_layer,
    _map_se,
    _map_svtr_key,
    _set_nested_attr,
)


class TestSetNestedAttr:
    def test_simple_attr(self):
        class Obj:
            pass

        o = Obj()
        _set_nested_attr(o, ["x"], 42)
        assert o.x == 42

    def test_nested_attr(self):
        class Inner:
            pass

        class Outer:
            def __init__(self):
                self.inner = Inner()

        o = Outer()
        _set_nested_attr(o, ["inner", "val"], "hello")
        assert o.inner.val == "hello"

    def test_digit_index(self):
        class Obj:
            def __init__(self):
                self.layers = [None, None, None]

            def __getitem__(self, idx):
                return self.layers[idx]

        class Container:
            def __init__(self):
                self.items = [Obj(), Obj()]

        c = Container()
        _set_nested_attr(c, ["items", "0", "val"], 99)
        assert c.items[0].val == 99

    def test_deep_nesting(self):
        class A:
            pass

        class B:
            def __init__(self):
                self.a = A()

        class C:
            def __init__(self):
                self.b = B()

        c = C()
        _set_nested_attr(c, ["b", "a", "value"], 3.14)
        assert c.b.a.value == 3.14


class TestIsLinearWeight:
    def test_qkv(self):
        assert _is_linear_weight("encoder.blocks.0.self_attn.qkv.weight") is True

    def test_proj(self):
        assert _is_linear_weight("encoder.blocks.0.self_attn.proj.weight") is True

    def test_fc1(self):
        assert _is_linear_weight("mlp.fc1.weight") is True

    def test_fc2(self):
        assert _is_linear_weight("mlp.fc2.weight") is True

    def test_fc(self):
        assert _is_linear_weight("head.fc.weight") is True

    def test_conv_is_not_linear(self):
        assert _is_linear_weight("backbone.conv1.weight") is False

    def test_bias_is_not_linear(self):
        assert _is_linear_weight("encoder.qkv.bias") is False


class TestMapConvBn:
    def test_conv_weight(self):
        assert _map_conv_bn("conv.weight") == "convolution.weight"

    def test_bn_weight(self):
        assert _map_conv_bn("bn.weight") == "normalization.weight"

    def test_bn_bias(self):
        assert _map_conv_bn("bn.bias") == "normalization.bias"

    def test_bn_mean(self):
        assert _map_conv_bn("bn._mean") == "normalization.running_mean"

    def test_bn_variance(self):
        assert _map_conv_bn("bn._variance") == "normalization.running_var"

    def test_norm_weight(self):
        assert _map_conv_bn("norm.weight") == "normalization.weight"

    def test_passthrough(self):
        assert _map_conv_bn("unrelated_key") == "unrelated_key"


class TestMapBn:
    def test_mean(self):
        assert _map_bn("_mean") == "running_mean"

    def test_variance(self):
        assert _map_bn("_variance") == "running_var"

    def test_weight_unchanged(self):
        assert _map_bn("weight") == "weight"


class TestMapSe:
    def test_conv1(self):
        assert _map_se("conv1.weight") == "convolutions.0.weight"

    def test_conv2(self):
        assert _map_se("conv2.bias") == "convolutions.2.bias"

    def test_passthrough(self):
        assert _map_se("other_key") == "other_key"


class TestMapRepLayer:
    def test_conv_kxk(self):
        result = _map_rep_layer("conv_kxk.0.conv.weight")
        assert result == "conv_symmetric.0.convolution.weight"

    def test_conv_kxk_bn(self):
        result = _map_rep_layer("conv_kxk.2.bn.weight")
        assert result == "conv_symmetric.2.normalization.weight"

    def test_conv_1x1(self):
        result = _map_rep_layer("conv_1x1.conv.weight")
        assert result == "conv_small_symmetric.convolution.weight"

    def test_bn_branch(self):
        result = _map_rep_layer("bn_branch._mean")
        assert result == "identity.running_mean"

    def test_lab(self):
        assert _map_rep_layer("lab.scale") == "lab.scale"
        assert _map_rep_layer("lab.bias") == "lab.bias"

    def test_act(self):
        assert _map_rep_layer("act.scale") == "act.scale"


class TestMapSvtrKey:
    def test_conv_blocks(self):
        assert _map_svtr_key("conv1.conv.weight") == (
            "head.encoder.conv_block.0.convolution.weight"
        )
        assert _map_svtr_key("conv1x1.bn.weight") == (
            "head.encoder.conv_block.1.normalization.weight"
        )
        assert _map_svtr_key("conv3.conv.weight") == (
            "head.encoder.conv_block.2.convolution.weight"
        )

    def test_svtr_block(self):
        result = _map_svtr_key("svtr_block.0.mixer.qkv.weight")
        assert result == "head.encoder.svtr_block.0.self_attn.qkv.weight"

    def test_svtr_norm(self):
        result = _map_svtr_key("svtr_block.0.norm1.weight")
        assert result == "head.encoder.svtr_block.0.layer_norm1.weight"

    def test_final_norm(self):
        result = _map_svtr_key("norm.weight")
        assert result == "head.encoder.norm.weight"


class TestMapPaddleKey:
    def test_skip_gtc(self):
        assert _map_paddle_key("head.gtc_head.fc.weight") is None
        assert _map_paddle_key("head.before_gtc.conv.weight") is None

    def test_stem_conv(self):
        result = _map_paddle_key("backbone.conv1.conv.weight")
        assert result == "model.backbone.encoder.convolution.convolution.weight"

    def test_ctc_head(self):
        result = _map_paddle_key("head.ctc_head.fc.weight")
        assert result == "head.head.weight"

    def test_svtr_encoder(self):
        result = _map_paddle_key("head.ctc_encoder.encoder.norm.weight")
        assert result == "head.encoder.norm.weight"

    def test_backbone_blocks(self):
        result = _map_paddle_key("backbone.blocks2.0.dw_conv.conv_kxk.0.conv.weight")
        assert "blocks.0.layers.0.depthwise_convolution" in result

    def test_unknown_key_returns_none(self):
        assert _map_paddle_key("totally.unknown.key") is None
