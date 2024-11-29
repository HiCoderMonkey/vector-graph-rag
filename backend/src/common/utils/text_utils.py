import re

from translate import Translator

# 创建翻译器对象
translator = Translator(from_lang="zh", to_lang="en")


def contains_chinese(text):
    # 检查文本是否包含中文字符
    return bool(re.search('[\u4e00-\u9fff]', text))


def translate_to_english(text):
    # 翻译文本
    translated = translator.translate(text)

    # 返回翻译后的文本
    return translated

def is_blank_string(s):
    return len(s.strip()) == 0
