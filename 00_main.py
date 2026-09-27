from openai import OpenAI
import json

client = OpenAI()

def validate_evaluation_criteria(state):
    criteria = state.get("evaluation_criteria")

    if not isinstance(criteria, list):
        print("\n=== データ構造エラー ===")
        print("evaluation_criteria は配列である必要があります。")
        raise SystemExit

    for number, item in enumerate(criteria, start=1):
        if not isinstance(item, dict):
            print(f"\n=== 評価基準 {number} の構造エラー ===")
            print("各評価基準はオブジェクトである必要があります。")
            raise SystemExit

        if not isinstance(item.get("criterion"), str):
            print(f"\n=== 評価基準 {number} の構造エラー ===")
            print("criterion は文字列である必要があります。")
            raise SystemExit

        if item.get("importance") not in ("必須", "重要", "参考"):
            print(f"\n=== 評価基準 {number} の構造エラー ===")
            print("importance は 必須・重要・参考 のいずれかです。")
            raise SystemExit

    return state

def filter_questions(state, context, system_prompt):
    if not state.get("questions"):
        return state

    question_filter_input = f"""
以下は、ユーザーの問題とAIが生成した追加質問候補です。

【ユーザーの入力・対話情報】
{context}

【現在の問題状態】
{json.dumps(state, ensure_ascii=False, indent=2)}

追加質問を厳しく審査してください。

質問は、
回答がないとユーザー固有の現実的な候補探索・比較ができず、
一般論や抽象的な条件付き提案しかできない場合に残してください。

ただし、
個別情報がなくてもユーザーが示した条件を使って
具体的な候補探索・比較を十分に進められる場合は削除してください。

例えば、

- 出発地・目的地が分からず、具体的な通勤経路を探索できない
- 本人の利用可否が分からず、主要な選択肢が成立するか判定できない

といった情報は質問として残して構いません。

一方で、

- 製品の仕様
- 価格
- 販売状況
- 候補ごとの差
- 好みの細分化

など、Web検索や条件付き比較で処理できる情報は質問しないでください。

次の質問は削除してください。

- 判断精度を少し上げるだけの質問
- 好みを細分化する質問
- 順位を微調整するための質問
- ユーザーが指定していない事項を新しい必須条件にする質問
- Web検索で候補側を調査できる質問
- 合理的な仮定を置けば進められる質問
- 条件付き評価として残せる質問

「回答によって候補が変わる可能性がある」
だけでは質問を残す理由になりません。

各質問について、
回答がなくても合理的な探索・比較・提案を開始できるなら
削除してください。

出力は、残す質問だけをJSON配列で返してください。

例：

["本当に必要な質問"]

すべて不要なら、

[]

としてください。

JSON以外は出力しないでください。
"""

    try:
        filter_response = client.responses.create(
            model="gpt-5.2",
            instructions=system_prompt,
            input=question_filter_input
        )
    except Exception as error:
        print("\n=== 質問フィルターAPI通信に失敗しました ===")
        print(f"エラー内容: {error}")
        raise SystemExit

    filtered_text = filter_response.output_text
    filtered_text = (
        filtered_text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        state["questions"] = json.loads(filtered_text)
    except json.JSONDecodeError as error:
        print("\n=== 質問フィルターJSONの読み込みに失敗しました ===")
        print(f"エラー内容: {error}")
        print("\nAIから返された内容:")
        print(filtered_text)
        raise SystemExit

    return state

# Project① v0.3を読み込む
with open("prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

print("=== 個人適応型AIエージェント Project① ===")

# -------------------------
# 1. 最初の入力
# -------------------------

goal = input("やりたいことを入力してください：")
request = input("要望を入力してください：")

user_input = f"""
やりたいこと：
{goal}

要望：
{request}

今回は通常の回答ではなく、
現在の問題状態を分析して、以下のJSON形式だけで出力してください。

{{
  "original_goal": "",
  "original_request": "",
  "problem_type": "",
  "current_state": "",
  "problem": "",
  "desired_state": "",
  "preferences": [],
  "constraints": [],
  "resources": [],
  "risks": [],
  "evaluation_criteria": [
  {{
    "criterion": "評価する内容",
    "importance": "必須・重要・参考のいずれか"
  }}
],
  "candidates": [],
  "missing_information": [],
  "questions": [],
  "user_answers": [],
  "recommendation": ""
}}

追加質問は原則0〜2問としてください。

質問するのは、
ユーザー本人にしか決められない情報であり、
かつ回答によって候補の除外・追加、解決方法、
上位候補の順位、最終結論、安全性のいずれかが
大きく変わる可能性が高い場合だけにしてください。

Web検索で確認できる仕様・価格・対応状況・販売状況などは
ユーザーに質問しないでください。

すでにユーザーが示している条件の言い換えや再確認、
単なる好みの細分化、
回答がなくても合理的に比較・判断できる情報は
質問しないでください。

質問候補ごとに、まず内部で、

「この回答がなくても、現在の情報、
合理的な仮定、後続のWeb検索、
または条件付き評価によって、
十分に良い候補探索・比較・提案を開始できるか」

を確認してください。

YESの場合は、その質問を削除してください。

NOの場合のみ、
その情報が必須条件、安全性、解決方法、
最終結論に重大な影響を与えるか確認し、
重大な影響がある場合だけ質問してください。

判断精度を少し上げるため、
好みをさらに細分化するため、
順位を微調整するためだけの質問はしないでください。

3問目を質問するのは、
安全性または必須条件の成立、
最終結論が大きく変わる可能性が非常に高い場合だけです。
"""

print("\nAIが問題を構造化しています...\n")

try:
    response = client.responses.create(
        model="gpt-5.2",
        instructions=system_prompt,
        input=user_input
    )
except Exception as error:
    print("\n=== API通信に失敗しました ===")
    print(f"エラー内容: {error}")
    raise SystemExit

json_text = response.output_text
json_text = json_text.replace("```json", "").replace("```", "").strip()

try:
    state = json.loads(json_text)
except json.JSONDecodeError as error:
    print("\n=== JSONの読み込みに失敗しました ===")
    print(f"エラー内容: {error}")
    print("\nAIから返された内容:")
    print(json_text)
    raise SystemExit

# 初期生成された評価基準の構造を検証
state = validate_evaluation_criteria(state)

# 質問候補を最終フィルター
state = filter_questions(
    state,
    user_input,
    system_prompt
)

# 初期状態を保存
with open("state.json", "w", encoding="utf-8") as file:
    json.dump(state, file, ensure_ascii=False, indent=2)

print("=== 初期状態を保存しました ===")


# -------------------------
# 2. 質問 → 回答 → 再評価ループ
# -------------------------

max_total_questions = 4
answered_question_count = 0
round_number = 1

while (
    state.get("questions")
    and answered_question_count < max_total_questions
):

    print(f"\n=== 追加質問 Round {round_number} ===")

    # 今回あと何問まで質問できるかを計算
    remaining_question_budget = (
        max_total_questions - answered_question_count
    )

    # AIが3問以上作っていても、
    # 全体の質問上限を超えない範囲だけ採用
    questions = state.get("questions", [])[
        :remaining_question_budget
    ]

    answers = []

    for number, question in enumerate(questions, start=1):

        print(f"\n質問{number}: {question}")
        answer = input("回答：")

        answers.append({
            "question": question,
            "answer": answer
        })

        answered_question_count += 1

    update_input = f"""
以下は現在の問題状態です。

{json.dumps(state, ensure_ascii=False, indent=2)}

ユーザーから追加回答が得られました。

{json.dumps(answers, ensure_ascii=False, indent=2)}

追加回答を反映して、
問題状態全体を再評価してください。

必要に応じて、
current_state、
problem、
desired_state、
preferences、
constraints、
resources、
risks、
evaluation_criteria、
candidates、
missing_information、
questions、
user_answers、
recommendation
を更新してください。

過去の user_answers は削除せず、
今回の回答を追加してください。

すでに回答済みの情報は、
missing_information や questions から削除してください。

追加質問は原則0〜2問としてください。

質問するのは、
ユーザー本人にしか決められない情報であり、
かつ回答によって候補の除外・追加、解決方法、
上位候補の順位、最終結論、安全性のいずれかが
大きく変わる可能性が高い場合だけにしてください。

Web検索で確認できる仕様・価格・対応状況・販売状況などは
ユーザーに質問しないでください。

すでに回答済みの情報や、
すでにユーザーが示している条件の言い換え・再確認、
単なる好みの細分化、
回答がなくても合理的に比較・判断できる情報は
質問しないでください。

質問候補ごとに、まず内部で、

「この回答がなくても、現在の情報、
合理的な仮定、後続のWeb検索、
または条件付き評価によって、
十分に良い候補探索・比較・提案を開始できるか」

を確認してください。

YESの場合は、その質問を削除してください。

NOの場合のみ、
その情報が必須条件、安全性、解決方法、
最終結論に重大な影響を与えるか確認し、
重大な影響がある場合だけ質問してください。

判断精度を少し上げるため、
好みをさらに細分化するため、
順位を微調整するためだけの質問はしないでください。

3問目を質問するのは、
安全性または必須条件の成立、
最終結論が大きく変わる可能性が非常に高い場合だけです。

十分な情報がある場合は、

"questions": []

としてください。

以下と同じJSON構造だけで出力してください。

{{
  "original_goal": "",
  "original_request": "",
  "problem_type": "",
  "current_state": "",
  "problem": "",
  "desired_state": "",
  "preferences": [],
  "constraints": [],
  "resources": [],
  "risks": [],
  "evaluation_criteria": [
  {{
    "criterion": "評価する内容",
    "importance": "必須・重要・参考のいずれか"
  }}
],
  "candidates": [],
  "missing_information": [],
  "questions": [],
  "user_answers": [],
  "recommendation": ""
}}
"""

    print("\nAIが回答を反映して再評価しています...\n")

    try:
        update_response = client.responses.create(
            model="gpt-5.2",
            instructions=system_prompt,
            input=update_input
        )
    except Exception as error:
        print("\n=== 再評価API通信に失敗しました ===")
        print(f"エラー内容: {error}")
        raise SystemExit

    updated_json_text = update_response.output_text
    updated_json_text = updated_json_text.replace(
        "```json", ""
    ).replace(
        "```", ""
    ).strip()

    try:
        state = json.loads(updated_json_text)
    except json.JSONDecodeError as error:
        print("\n=== 再評価JSONの読み込みに失敗しました ===")
        print(f"エラー内容: {error}")
        print("\nAIから返された内容:")
        print(updated_json_text)
        raise SystemExit

    # 再評価後の評価基準の構造を検証
    state = validate_evaluation_criteria(state)

    # 再評価後の質問候補も最終フィルター
    state = filter_questions(
        state,
        update_input,
        system_prompt
    )

    # 更新状態を保存
    with open("state.json", "w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nstate.json を更新しました。")

    print(
        f"質問数: "
        f"{answered_question_count}/{max_total_questions}"
    )

    round_number += 1

# -------------------------
# 3. 最終結果
# -------------------------

print("\n=== 最終問題状態 ===")
print(json.dumps(state, ensure_ascii=False, indent=2))

print("\n=== Project①の提案 ===")
print(state.get("recommendation", "提案はまだありません。"))

if (
    state.get("questions")
    and answered_question_count >= max_total_questions
):
    print(
        "\n※ 質問上限に達したため、"
        "残りの不足情報は保留してWeb探索へ進みます。"
    )


# -------------------------
# 4. Web検索
# -------------------------

# Web検索が必要か判定
web_judge_input = f"""
以下は現在の問題状態です。

{json.dumps(state, ensure_ascii=False, indent=2)}

この問題を解決するために、
最新または外部のWeb情報が必要か判定してください。

Web検索が必要な例：
- 現在販売されている製品の比較
- 最新価格・仕様・在庫・制度の確認
- 店舗・交通・サービスなど実在情報の確認
- 最新情報が判断を左右する問題

Web検索が不要な例：
- 文章・物語・アイデアの生成
- 与えられた情報だけで完結する整理
- 一般的な方法や考え方の設計

必要なら

YES

不要なら

NO

だけを出力してください。
"""

try:
    web_judge_response = client.responses.create(
        model="gpt-5.2",
        input=web_judge_input
    )
except Exception as error:
    print("\n=== Web検索判定API通信に失敗しました ===")
    print(f"エラー内容: {error}")
    raise SystemExit

needs_web = (
    web_judge_response.output_text.strip().upper() == "YES"
)

print(f"\nWeb検索の必要性: {needs_web}")

if not needs_web:
    print("\n=== Web検索は不要と判定しました ===")
    print("\n=== Webなしで最終回答を生成します ===")

    no_web_final_input = f"""
以下は、ユーザーとの対話によって整理された最終問題状態です。

【現在の問題状態】
{json.dumps(state, ensure_ascii=False, indent=2)}

Web検索はこの問題には不要と判定されています。

現在の問題状態だけを使って、
ユーザーの「やりたいこと」と「要望」を実際に達成する
完成した最終回答を作成してください。

重要：
recommendation に書かれている途中方針を
そのまま繰り返すだけで終わらないでください。

ユーザーが求めている成果物・解決策・候補・方法を、
現在ある情報だけで可能な範囲まで実際に作成してください。

不足情報があっても、
合理的な仮定または条件付きの提案で進められる場合は、
追加質問をせずに完成回答を提示してください。

例えば、

- 小説案を求めているなら、実際の小説案を提示する
- アイデアを求めているなら、実際のアイデアを提示する
- 方法を求めているなら、実行できる方法を提示する
- 整理を求めているなら、整理した結果を提示する

という形にしてください。

このフェーズでは、
ユーザーへの追加質問を行わないでください。

必要に応じて未確定事項や仮定を明示しつつ、
現在の情報だけで最善の完成回答を作成してください。
"""

    try:
        no_web_final_response = client.responses.create(
            model="gpt-5.2",
            instructions=system_prompt,
            input=no_web_final_input
        )
    except Exception as error:
        print("\n=== 最終回答API通信に失敗しました ===")
        print(f"エラー内容: {error}")
        raise SystemExit

    print("\n=== Project① 最終提案 ===")
    print(no_web_final_response.output_text)

    raise SystemExit

print("\n=== Web検索を実行します ===")

web_input = f"""
以下は、ユーザーとの対話によって整理された最終問題状態です。

{json.dumps(state, ensure_ascii=False, indent=2)}

この問題について、現在入手可能な情報をWeb検索し、
ユーザーに適した候補・解決方法そのものを探索してください。

重要：
現在の state に含まれる candidates は、
Web検索前にAIの知識から作られた暫定候補です。

その候補だけを検索・検証してはいけません。

まず、
目的・希望・必須条件・制約・利用条件・評価基準
を基準として、現在存在する候補をWebから広く探索してください。

そのうえで、

1. 必須条件を満たす候補を抽出する
2. 暫定候補以外に適した候補があれば追加する
3. 現在の情報では条件を満たさない候補を除外する
4. 判断に重要な最新の価格・仕様・対応状況などを確認する
5. 有力候補をユーザーの評価基準で比較する

という順で調査してください。

検索前の candidates や recommendation に固執しないでください。
Web検索の結果によって候補を全面的に入れ替えて構いません。

可能な限り、
メーカー公式、公式仕様、公式販売ページなどの
一次情報を優先してください。

Web検索で確認できた事実と、
推測・一般知識を明確に区別してください。

確認できなかった情報は、
確認済みとして扱わないでください。

特に、必須条件については確認を優先してください。

有力候補について必須条件の確認が取れていない場合、
すぐに「未確認」として探索を終了しないでください。

【UNKNOWN判定前の追加探索】

必須条件をUNKNOWNとする前に、
その条件が公式仕様・公式FAQ・公式販売ページなどで
確認できる情報かを検討してください。

特に、メーカー名・機種名・型番・容量を組み合わせて
追加検索してください。

一度の検索で情報が見つからなかったことを、
UNKNOWNの十分な理由としてはいけません。

ただし、別の機種・容量・販売形態の情報を
対象候補の確認根拠として流用してはいけません。

複数の適切な情報源を調べても確認できなかった場合は、
推測でPASSにせずUNKNOWNとしてください。

UNKNOWNとした場合は、
何が確認できなかったのかを具体的に記載してください。

その情報がWeb上で確認可能な事実である場合は、
検索語や参照先を変えて追加探索し、
可能な範囲で確認してください。

例えば、

- メーカー公式仕様ページ
- メーカー公式FAQ・サポートページ
- 公式販売ページ
- 通信事業者の公式仕様ページ
- 信頼できる販売事業者の仕様ページ

など、別の情報源も探索してください。

特に、

- 価格
- 容量
- 対応機能
- 防水防塵
- 通信対応
- 販売状態
- 新品・SIMフリーで購入可能か

など、Webで確認可能な必須条件は、
有力候補ごとに確認を試みてください。

必須条件が未確認のまま残る場合は、
その条件を確認するための追加探索を行ったうえで、
それでも確認できなかった場合にのみ
「未確認」としてください。

また、有力候補ごとに必須条件を個別に照合してください。

各候補について、

- 新品で購入可能か
- SIMフリーか
- Androidか
- 予算上限以内で購入可能か
- 必要なストレージ容量を満たすか
- おサイフケータイに対応するか
- 必要な防水防塵条件を満たすか
- 必要な通信回線で利用可能か

など、そのユーザーが指定した必須条件を
1項目ずつ確認してください。

必須条件の一部だけを確認できた候補を、
「必須条件を満たすことを確認できた候補」
に分類してはいけません。

「主要な必須条件を確認できた」
「概ね条件を満たす」
「国内モデルなので対応していると思われる」
などを、
必須条件の完全確認の代わりに使用しないでください。

すべての必須条件について確認済みの根拠がある場合だけ、
「必須条件確認済み」としてください。

必須条件の完全確認は、候補ごとのチェック結果だけで判定してください。

ある候補について必須条件をN個設定した場合、
N個すべてが「確認済み」である場合に限り、
その候補を「必須条件確認済み」としてください。

1項目でも「未確認」「推定」「可能性が高い」
「公式には完全確認できていない」が残る場合は、
その候補全体を「必須条件未確認候補」としてください。

例えば、
8個の必須条件のうち7個を確認できても、
残り1個が未確認なら、
「必須条件確認済み」ではありません。

「ドコモが取り扱っているため利用できる可能性が高い」
「国内モデルなので対応していると考えられる」
「相場的に予算内と思われる」
などの推論は、
必須条件の確認済み判定には使用しないでください。

【Web探索結果の出力順序と整合性】

【出力順序の厳守】

Web探索結果は、必ず次の順序で出力してください。

1. 必須条件の一覧
2. 候補ごとのPASS / FAIL / UNKNOWN判定表
3. 各候補のPASS / FAIL / UNKNOWN集計
4. 集計に基づく候補分類
5. 結論・候補比較・未確認事項

判定表と集計を完成させる前に、
冒頭の結論・推薦機種・確認済み候補数を書いてはいけません。

冒頭に「結論」「要約」「おすすめ」などの項目を設けず、
必ず必須条件の一覧から出力を開始してください。

候補分類は、完成した判定表の集計結果だけを根拠に決定してください。

まず各候補について、ユーザーが指定した必須条件を
1項目ずつPASS / FAIL / UNKNOWNで判定してください。

すべての候補の判定表と集計を完成させてから、
確認済み候補・未確認候補・不適合候補を分類してください。

冒頭の結論や候補の推薦は、
完成した判定表と集計に基づいて記載してください。

判定表ではUNKNOWNが残っているのに、
冒頭やまとめで「全条件PASS」と記載してはいけません。

文章全体で、候補の分類とPASS / FAIL / UNKNOWNの
集計結果を一致させてください。


判定基準：

PASS：
その必須条件を満たすことが、
具体的な情報源によって確認できた場合。

FAIL：
その必須条件を満たさないことが、
具体的な情報源によって確認できた場合。

UNKNOWN：
情報が不足している場合、
確認できた情報だけでは条件充足を判断できない場合、
または推測によってしか判断できない場合。

【必須条件の追加・厳格化の禁止】

必須条件は、ユーザーが実際に指定した内容を基準に判定してください。

ユーザーが指定していない性能・規格・数値・等級を、
AIの判断で新たな必須条件として追加してはいけません。

例えば、ユーザーが「防水防塵」とだけ指定した場合、
公式仕様で防水防塵対応を確認できればPASSとしてください。

ユーザーがIP68などの具体的な等級を指定していない限り、
IP等級を確認できないという理由だけでUNKNOWNにしてはいけません。

ただし、公式仕様などで防水防塵への対応自体を
確認できない場合はUNKNOWNとしてください。

ユーザーが指定していない詳細性能は、
必要に応じて比較項目・参考情報・注意事項として扱ってください。

特に価格条件については、
新品の具体的な税込販売価格を確認し、
ユーザーの予算上限と照合してください。

新品の販売ページが存在すること、
価格比較ページが存在すること、
一般的な相場から予算内と推測できること、
発売時の価格が予算内だったことだけでは、
現在の価格条件をPASSにしてはいけません。

価格条件をPASSにする場合は、
確認できた税込価格・販売元・確認時点を示してください。

販売価格を確認できなかった場合はUNKNOWNとしてください。

【現在価格の確認ルール】

【価格・購入可能性の根拠の整合性】

現在価格のPASS判定には、
調査時点で実際に購入できる対象商品の個別販売ページを使用してください。

発売時の公式発表価格や過去の記事に記載された価格を、
現在価格のPASS根拠として使用してはいけません。

価格と新品購入可能性は、原則として同じ販売元・
同じ型番・同じ容量・同じ販売形態の情報で照合してください。

例えば、メーカーの過去の発表価格と、
別の販売店の在庫情報を組み合わせて、
現在その価格で購入できると判断してはいけません。

販売ページで現在の税込価格を確認できても、
在庫切れ・販売終了などで購入できない場合は、
現在購入可能な価格としてPASSにしてはいけません。

ある販売店で予算超過の価格を確認しただけでは、
他の販売店も含めて予算内の購入先が存在しないとは断定できません。
別の適切な販売先も探索し、予算内で購入できる根拠が
確認できなければ、価格条件はUNKNOWNとしてください。

価格と新品購入可能性をPASSにする場合は、
根拠となる販売元・商品ページ・税込価格・購入可能状態を
互いに矛盾なく提示してください。

価格条件は、調査時点で購入できる新品の
具体的な税込販売価格を確認した場合のみPASSとしてください。

発売時の希望小売価格、過去のプレスリリース、
終了済みキャンペーン、過去のセール記事に掲載された価格は、
現在の購入価格を証明する根拠として使用しないでください。

現在の販売ページで、対象の型番・容量・新品状態・
税込価格・販売元・購入可能な状態を確認してください。

確認できた場合は、価格と販売元、
確認日、参照した販売ページを記載してください。

過去の価格しか確認できなかった場合、
現在の販売価格が確認できない場合、
在庫や購入可能な状態が不明な場合はUNKNOWNとしてください。

価格条件のPASSと、新品で購入可能という条件のPASSは
それぞれ独立して判定してください。

価格比較サイトの最安価格・価格レンジ・販売店一覧は、
実際の販売店で購入可能であることの証明にはなりません。

価格比較サイトだけを根拠に、
現在価格や新品購入可能性をPASSにしてはいけません。

価格比較サイトで候補を発見した場合は、
掲載されている販売店の個別商品ページを追加確認してください。

個別販売店で対象商品・新品状態・税込価格・購入可能状態を
確認できなければ、該当する条件はUNKNOWNとしてください。

候補全体の判定は、次のルールに統一してください。

・すべての必須条件がPASS
  → 必須条件確認済み候補

・1項目でもFAILがある
  → 必須条件不適合候補

・FAILがなく、1項目以上のUNKNOWNがある
  → 必須条件未確認候補

UNKNOWNをPASSとして扱ってはいけません。

また、FAILがある候補についても、
UNKNOWNを確認済みと表現してはいけません。

各候補について、
必須条件・判定結果・確認根拠を一覧表で整理してください。

必須条件がN個ある場合は、
PASS数 / FAIL数 / UNKNOWN数を集計し、
合計が必ずN個になるようにしてください。

1項目でも確認できない必須条件が残る場合は、
その項目について追加探索してください。

追加探索しても確認できない場合は、
候補自体を除外する必要はありませんが、
「必須条件未確認候補」として明確に分離してください。

必須条件を確認済みとする場合は、
どの必須条件をどの情報源で確認したかが
追跡できる形で整理してください。

ただし、同じ情報を無制限に検索し続けないでください。
複数の適切な情報源を確認しても判断できない場合は、
未確認として探索を終了してください。

最後に、

・Webから新たに発見した有力候補
・必須条件を満たすことを確認できた候補
・除外すべき候補とその理由
・候補比較に重要な確認済み事実
・まだ確認できていない重要情報

を簡潔に整理してください。

このWeb探索フェーズでは、
ユーザーへの追加質問を行わないでください。

不足情報が残っている場合は、
まずWeb検索によって確認できないか探索してください。

Web検索でも確認できない情報や、
ユーザー本人にしか決められない情報については、
質問としてユーザーへ返さず、
「まだ確認できていない重要情報」として整理してください。

不足情報が残っていても、
現在得られている情報だけで可能な範囲まで
探索・比較を進めてください。
"""

try:
    web_response = client.responses.create(
        model="gpt-5.2",
        tools=[
            {
                "type": "web_search"
            }
        ],
        input=web_input
    )
except Exception as error:
    print("\n=== Web検索API通信に失敗しました ===")
    print(f"エラー内容: {error}")
    raise SystemExit

print("\n=== Web検索結果 ===")
print(web_response.output_text)

# -------------------------
# 4.5 Web検索結果の独立検証
# -------------------------

print("\n=== Web検索結果の独立検証を開始します ===")

verification_input = f"""
以下は、ユーザーの問題状態とWeb探索結果です。

【問題状態】
{json.dumps(state, ensure_ascii=False, indent=2)}

【Web探索結果】
{web_response.output_text}

【検証指示】
Web探索結果を、元の探索担当とは独立した立場で監査してください。

特に次を確認してください。
1. 必須条件がユーザーの指定と一致し、勝手に追加・厳格化されていないか。
2. 判定表の項目数とPASS・FAIL・UNKNOWNの集計が一致しているか。
3. 候補分類が各項目の判定結果と一致しているか。
4. 現在価格のPASSに、対象商品の個別販売ページ・税込価格・新品状態・購入可能状態の根拠があるか。
5. 発売時の価格、過去の記事、価格比較ページ、別容量の商品情報を現在価格の根拠に流用していないか。
6. 根拠不足の項目を推測でPASSにしていないか。

問題があれば、該当候補・該当条件・問題点・必要な修正を具体的に示してください。
根拠が不足する場合はPASSを維持せず、UNKNOWNへの修正を指示してください。

"""

try:
    verification_response = client.responses.create(
        model="gpt-5.2",
        input=verification_input
    )
except Exception as error:
    print("\n=== 独立検証API通信に失敗しました ===")
    print(f"エラー内容: {error}")
    raise SystemExit

print("\n=== Web検索結果の独立検証結果 ===")
print(verification_response.output_text)

# -------------------------
# 5. Web検索結果を使った最終再評価
# -------------------------

print("\n=== Web検索結果を反映して最終再評価します ===")

final_input = f"""
以下は、ユーザーとの対話によって整理された問題状態です。

【現在の問題状態】
{json.dumps(state, ensure_ascii=False, indent=2)}

以下は、その問題についてWeb検索で確認した最新情報です。

【Web検索結果】
{web_response.output_text}

【独立検証結果】
{verification_response.output_text}

最終回答では、Web検索結果だけでなく独立検証結果も反映してください。
独立検証で指摘された集計ミス・分類ミス・根拠不足を修正してください。
根拠不足でUNKNOWNへの変更を指示された項目を、追加根拠なしにPASSへ戻してはいけません。

現在の問題状態とWeb検索結果の両方を使って、
ユーザーに最も適した最終判断を行ってください。

特に以下を守ってください。

【必須条件の最終判定ルール】

Web探索結果の必須条件判定を引き継いでください。

各候補の必須条件は、
PASS / FAIL / UNKNOWNの3段階で扱ってください。

PASS：
具体的な根拠によって条件を満たすと確認できた場合。

FAIL：
具体的な根拠によって条件を満たさないと確認できた場合。

UNKNOWN：
情報不足、根拠不足、推測によってしか判断できない場合。

Web探索結果にUNKNOWNが残っている条件を、
新しい根拠なしにPASSへ変更してはいけません。

特に価格条件については、
具体的な新品の税込販売価格・販売元・確認時点が
確認できていなければUNKNOWNとしてください。

候補全体の分類は以下に統一してください。

・全必須条件PASS → 必須条件確認済み候補
・1項目以上FAIL → 必須条件不適合候補
・FAILなし、UNKNOWNあり → 必須条件未確認候補

すべての必須条件がPASSの候補だけを、
必須条件確認済みとして推薦してください。

必須条件未確認候補は、
条件付き候補・確認待ち候補として明確に区別してください。

必須条件確認済み候補が1件もない場合は、
その事実を明示してください。
未確認候補を確認済み候補の代わりに
確定的に推薦してはいけません。

・ユーザーの目的、希望、条件、制約、評価基準を優先する
・Web検索で確認できた事実を最終判断に反映する
・検索前の候補や順位に固執しない
・Web検索によって候補を除外、追加、順位変更してよい
・確認できなかった情報は、確認済みとして扱わない
・事実と推論を区別する
・不足情報が残っていて順位を確定できない場合は、無理に確定しない
・必須条件について未確認の項目が1つでも残る候補は、「必須条件を満たすことが確認済み」と扱わない
・必須条件が1つでも未確認の候補は、「第一候補」「本命」「最終1位」「最も適している」など、必須条件を満たすことを前提とした確定的な表現で推薦しない
・必須条件をすべて確認できた候補がある場合は、原則として未確認の候補より優先する
・有力候補の必須条件が未確認の場合は、「条件付き候補」または「確認待ち」として扱う
・この最終再評価フェーズでは、ユーザーへの追加質問を行わない
・ユーザー本人にしか分からない情報が不足していても、質問として返さない
・不足情報は「未確認・注意事項」として明示する
・不足情報があっても、現在得られている情報の範囲で最善の判断を行う

今回はJSONではなく、
ユーザーがそのまま読める最終提案として出力してください。

以下の順で簡潔にまとめてください。

1. 結論
2. 主な理由
3. 候補比較
4. 未確認・注意事項
"""

try:
    final_response = client.responses.create(
        model="gpt-5.2",
        instructions=system_prompt,
        input=final_input
    )
except Exception as error:
    print("\n=== 最終再評価API通信に失敗しました ===")
    print(f"エラー内容: {error}")
    raise SystemExit

print("\n=== Project① Web反映後の最終提案 ===")
print(final_response.output_text)