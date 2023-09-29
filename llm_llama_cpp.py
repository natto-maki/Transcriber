import os
import subprocess
import logging

# noinspection PyPackageRequirements
from llama_cpp import Llama

# for summarize
_model_file_name = "ggml-model-q4_m.gguf"
_model_file_url = "https://huggingface.co/TFMC/openbuddy-llama2-13b-v11.1-bf16-GGUF/resolve/main/ggml-model-q4_m.gguf"

# for translation (Low success rate...)
#_model_file_name = "llama-2-7b-chat.Q5_0.gguf"
#_model_file_url = "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_0.gguf"

def _check_dependency():
    if not os.path.isfile(_model_file_name):
        r0 = subprocess.run(["wget", _model_file_url])
        if r0.returncode != 0:
            raise RuntimeError("Cannot download model file - %s" % _model_file_url)


_check_dependency()
_model = Llama(model_path=_model_file_name, n_ctx=2048, n_gpu_layers=100)


def _invoke(messages, model_name: str | None = None):
    _ = model_name  # TODO model_name -> _model instance

    prompt_list = []
    for m in messages:
        role = m["role"]
        if role == "system":
            prompt_list.append(m["content"])
        elif role == "user":
            prompt_list.append("USER: " + m["content"])
        elif role == "assistant":
            prompt_list.append("ASSISTANT: " + m["content"])
        else:
            raise ValueError()
    prompt_list.append("ASSISTANT: ")

    r0 = _model.create_completion(
        "\n".join(prompt_list),
        temperature=0.7, top_p=0.3, top_k=40, repeat_penalty=1.1, max_tokens=1024,
        stop=["ASSISTANT:", "USER:", "SYSTEM:"],
        stream=False)

    print("\n".join(prompt_list))
    print(r0)

    finish_reason = r0["choices"][0]["finish_reason"]
    if finish_reason != "stop":
        raise RuntimeError("LLM finished with unexpected reason: " + finish_reason)

    return r0["choices"][0]["text"]


# https://github.com/abetlen/llama-cpp-python
# https://huggingface.co/TFMC
#
# sudo apt install -y cmake clang lldb lld wget
#
# ubuntu+GPU
#   CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
# x86 Mac (very slow)
#   CMAKE_ARGS="-DLLAMA_METAL=off" pip3 install llama-cpp-python

test_prompt1 = '''\
Please translate it into clean English. Input text is in Japanese.
'''

test_prompt2 = '''\
Please translate it into clean Japanese. Input text is in English.
'''

test_prompt3 = '''\
The following text is the transcribed minutes of a conversation during a meeting.
From this transcript, please extract a summary and action items.
The summary should include only proper nouns or the content of the discussion,
and should not be supplemented with general knowledge or known facts.
Action items should only include items explicitly mentioned by participants in the agenda, 
and should not include speculation.
The action item should be prefixed with "Action item:".
For example, the format is as follows:
要点を冒頭に出力します。文章にしてください。
Action item: アクションアイテムの例
Action item: アクションアイテムは複数になることもあります。
":" 以降は日本語にしてください。ただし、日本語表記ではない人名は日本語に変換せず、原表記を維持してください。
If there is no particular information to be output, or if there is not enough information for the summary,
just output "none".
'''

test_prompt4 = '''\
From this transcript, please extract a short digest.
日本語で出力してください。
'''

test_prompt5 = '''\
Extract only the main points from the following text and generate a short digest.
'''

test_prompt6 = '''\
What is the main message in this conversation?
'''

test_text1 = '''\
ドライバーが商品をお届けに向かっています。
注文時にお届け先の位置を修正した場合、修正前の位置が表示されることがあります。
'''

test_text2 = '''\
The driver is delivering the product. 
If the delivery address was modified during the order process, the previous address may be displayed.
'''

test_text3 = '''\
スペイン継承戦争を終結させた1713年のユトレヒト条約および1714年のラシュタット条約により、
スペインはネーデルラント、ミラノ公国、ナポリ王国、サルデーニャ島を神聖ローマ皇帝カール6世に、
シチリア王国をサヴォイア公ヴィットーリオ・アメデーオ2世に割譲した。
スペイン王フェリペ5世は領土回復を追求したが、当面の急務は13年間の戦争で荒廃したスペインの国力の回復であり、
イタリア出身の枢機卿であるジュリオ・アルベローニ（英語版）がそれを推進した。
1714年、アルベローニは寡夫となったフェリペ5世と21歳のエリザベッタ・ファルネーゼの縁談をまとめた。
この縁談の途中でアルベローニはエリザベッタの個人的な顧問になった。
1715年、アルベローニは首相に就任、スペインの財政と陸軍を改革したほかスペイン艦隊を再建した
（1718年だけで戦列艦を50隻建造した）。
一方のエリザベッタもファルネーゼ家の一員としてイタリアのパルマ・ピアチェンツァ公国、
ひいてはトスカーナ大公国の継承権を有していたため、自分の子供であるドン・カルロス王子のためにイタリアの君主位を確保したいと望み、
アルベローニの支持を得てフェリペ5世とその息子たちのイタリアに対する野心を煽った。
そして、スペインはオーストリアがオスマン帝国との戦争（墺土戦争）にかかりきりになっている間に、
軍隊を派遣して1717年8月にサルデーニャを占領した（スペインによるサルデーニャ侵攻）。
'''

test_text4 = '''\
それじゃあ、私が調べた「高齢猫の注意点」について説明しますね。

猫は年を取るにつれて、1日のほとんどを眠って過ごすようになるみたいですね。
特に寒い冬は暖かい場所を好んで昼寝したりして、そこでずっと1日を過ごすことが多いみたいですね。
で、飼い主が気を付けなくてはいけないのが低温やけどです。

低温やけどを防ぐためにも、猫がコタツみたいな暖房器具の近くにいる時は、温度を調節したり、
長時間同じ場所で過ごさないように、気を付けてあげることが重要みたいですね。
だけど、温度が下がりすぎることで具合を悪くしてしまう高齢猫もいるみたいで、なので、飼い主が家を出ている時は「湯たんぽ」がお勧めです。
'''

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
logging.info("result = \"%s\"" % _invoke(messages=[
    {"role": "system", "content": test_prompt6},
    {"role": "user", "content": test_text4}
]))
