import json
import os
import jieba

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
LEXICON_PATH = os.path.join(BASE_DIR, "data/processed/step5/shared/bilibili_sentiment_lexicon.json")

DEFAULT_POS = [
    "牛逼", "绝了", "神了", "好用", "厉害", "强", "赞", "棒", "优秀", "精彩",
    "干货", "学到了", "感谢", "清晰", "详细", "爱", "666", "稳", "妙", "不错"
]
DEFAULT_NEG = [
    "离谱", "就这", "不行", "垃圾", "劝退", "浪费时间", "失望", "无聊", "差", "水",
    "营销号", "骗子", "忽悠", "烂", "坑", "无语", "尴尬", "假"
]
DEFAULT_ANXIETY = [
    "焦虑", "失业", "取代", "怎么办", "危机", "淘汰", "卷", "内卷", "恐慌", "凉凉",
    "完蛋", "被替代", "威胁", "红利结束", "寒冬", "裁员", "降薪", "没前途"
]

def ensure_lexicon():
    if os.path.exists(LEXICON_PATH):
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    lexicon = {"pos": DEFAULT_POS, "neg": DEFAULT_NEG, "anxiety": DEFAULT_ANXIETY}
    os.makedirs(os.path.dirname(LEXICON_PATH), exist_ok=True)
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    return lexicon

_lexicon = None

def load_lexicons():
    global _lexicon
    if _lexicon is None:
        _lexicon = ensure_lexicon()
    return _lexicon

def _tokenize(text):
    if not isinstance(text, str):
        return []
    return list(jieba.cut(text.strip()))

def score_sentiment(text):
    lex = load_lexicons()
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos_set = set(lex["pos"])
    neg_set = set(lex["neg"])
    pos_count = sum(1 for t in tokens if t in pos_set)
    neg_count = sum(1 for t in tokens if t in neg_set)
    total = len(tokens)
    return (pos_count - neg_count) / total

def score_anxiety(text):
    lex = load_lexicons()
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    anxiety_set = set(lex["anxiety"])
    anxiety_count = sum(1 for t in tokens if t in anxiety_set)
    return anxiety_count / len(tokens)

def batch_score(df, content_col="content"):
    df = df.copy()
    df["sentiment"] = df[content_col].apply(score_sentiment)
    df["anxiety"] = df[content_col].apply(score_anxiety)
    return df
