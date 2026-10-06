"""台語連讀變調 v1（F1 漳州腔）

規則：同一連字號詞內，除最後一個音節外都變調；'--' 之後為輕聲，標為 0
不處理：跨詞變調、仔前變調、三疊字、第 6/9 聲

模型 B 的訓練資料經過本轉換，所以推論時的輸入也必須經過同一個轉換。
（由 cb_infer.TaigiTTS 自動處理，呼叫端只要送本調台羅）
"""
import re

_RULE = {'1': '7', '2': '1', '3': '2', '5': '7', '7': '3'}
_SYL  = re.compile(r'^([A-Za-z]+)([0-9])$')
_WORD = re.compile(r'[A-Za-z0-9]+(?:-{1,2}[A-Za-z0-9]+)*')


def sandhi_syl(s):
    m = _SYL.match(s)
    if not m:
        return s
    base, t = m.groups()
    if t in _RULE:
        return base + _RULE[t]
    if t == '4':                                   # 陰入：-h→2，-p/t/k→8
        return base + ('2' if base[-1].lower() == 'h' else '8')
    if t == '8':                                   # 陽入：-h→3，-p/t/k→4
        return base + ('3' if base[-1].lower() == 'h' else '4')
    return s                                       # 6、9 不處理


def neutral_syl(s):
    m = _SYL.match(s)
    return m.group(1) + '0' if m else s


def convert_word(w):
    parts = w.split('--')
    head = parts[0].split('-') if parts[0] else []
    if head:
        head = [sandhi_syl(x) for x in head[:-1]] + head[-1:]
    out = '-'.join(head)
    for p in parts[1:]:
        out += '--' + '-'.join(neutral_syl(x) for x in p.split('-'))
    return out


def convert(text):
    return _WORD.sub(lambda m: convert_word(m.group(0)), text)


if __name__ == '__main__':
    tests = {
        'kho1-ki1':            'kho7-ki1',
        'lai7-kho1':           'lai3-kho1',
        'kua3-ho7':            'kua2-ho7',
        'hak8-sing1':          'hak4-sing1',
        'tsheh4-png5':         'tsheh2-png5',
        'khuann3--khi2':       'khuann3--khi0',
        'peh4--a2':            'peh4--a0',
        'thau5-khak4 thiann3': 'thau7-khak4 thiann3',
    }
    ok = 0
    for i, exp in tests.items():
        got = convert(i); ok += got == exp
        print(f"{'✓' if got == exp else '✗'} {i:22s} → {got:22s} 預期 {exp}")
    print(f"{ok}/{len(tests)} 通過")
