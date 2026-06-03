import sys

conv_dict = {
  "ami": "Amis",
  "bnn": "Bunun",
  "ckv": "Kavalan",
  "dru": "Rukai",
  "pwn": "Paiwan",
  "ssf": "Thao",
  "sxr": "Saaroa",
  "szy": "Sakizaya",
  "tao": "Yami",
  "tay": "Atayal",
  "trv": "Truku",
  "sqf": "Seediq",  # this code is not part of ISO 639-3, it is not assigned there as of 2026-05-25 - we have separate trv/sqf datasets, so I add this code here
  "tsu": "Tsou",
  "xnb": "Kanakanavu",
  "xsy": "Saisiyat",
  "pyu": "Puyuma",
}

inv_dict = {v: k for k, v in conv_dict.items()}

LANG_CODES = list(conv_dict.keys())
LANG_NAMES = list(inv_dict.keys())

def get_langs(codes):
  return [conv_dict.get(code, f"{code} not found") for code in codes]


def get_codes(langs):
  return [inv_dict.get(lang, f"{lang} not found") for lang in langs]


def main():
  if len(sys.argv) < 2:
    print("Usage: python3 convert_lang_code.py [-c/--get-code/-l/--get-lang] [lang1 lang2 lang3 .../code1 code2 code3 ...]")

  dir = sys.argv[1]

  use_dict = conv_dict if dir in ["-l", "--get-lang"] else inv_dict 

  for lang in sys.argv[2:]:
    print(f"{lang}: {use_dict.get(lang, f'{lang} not found')}")


if __name__ == "__main__":
  main()
