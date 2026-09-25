"""Parser for C/C++ weight initializers used by the original firmware."""
import re

def parse_numeric_initializer(text):
    """Extract integer/float literals from a C++ initializer.

    Returns a flat list. This intentionally preserves source ordering; callers
    reshape according to the declared dimensions in the header.
    """
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    body = text[text.find('{'):] if '{' in text else text
    toks = re.findall(r'(?<![A-Za-z_])[+-]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][+-]?\d+)?', body)
    return [float(t) for t in toks]

def quantize_ternary(values, threshold=0.5):
    return [1 if v>=threshold else -1 if v<=-threshold else 0 for v in values]

def load_header(path, ternary=False):
    vals=parse_numeric_initializer(open(path, encoding='utf-8', errors='ignore').read())
    return quantize_ternary(vals) if ternary else vals

def load_original_header(path):
    """Return the repository's exact 16 conv filters and FC tensor.

    The header stores 16x16 ternary conv values and 10x16x16x16 FC values.
    The latter is laid out neuron-major, then feature-map, then 16x16 pooled
    coordinates, matching load_FC_weights_into_grid_E().
    """
    vals=[int(v) for v in load_header(path)]
    if len(vals) != 16*16 + 10*16*16*16:
        raise ValueError(f'expected 41216 weight values, found {len(vals)}')
    conv=[vals[i*16:(i+1)*16] for i in range(16)]
    fc=vals[256:]
    return conv, fc
