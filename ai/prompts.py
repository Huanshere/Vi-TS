## ========== @ ai.analyze_video_302.py ==========
def get_analyze_clothing_prompt():
    prompt = """
## Content 
你是一个专业的图片分析专家，专门分析图片中人员在室内热环境中的衣着情况。你的任务是详细描述图片中人员的上衣和裤子的种类、材质、覆盖面积，以及相关的补充信息（包括帽子、围巾和椅子接触面积等会影响服装热阻的因素）。

## Task
分析图片中人员的衣着，并以 JSON 格式返回分析结果。如果服装被遮挡，例如裤子被桌子挡住，你需要进行适当的估计。

## Action
1.  **整体观察：** 观察图片，只需要分析最主要的人物，以及大致的室内环境。
2.  **衣着分析：** 针对此人，分析其衣着。
    *   **上衣分析：**
        *   **种类 (upper_type)：** 衬衫、T恤、毛衣、外套、卫衣等。
        *   **材质 (upper_material)：** 棉、麻、丝绸、涤纶、羊毛、针织等（根据视觉判断）。
        *   **覆盖面积 (upper_coverage)：** 估计上衣覆盖身体的比例（例如，覆盖上半身，覆盖手臂等）。
        *   **细节 (upper_details)：** 是否有领子、袖子长度、是否敞开等。
    *   **裤子分析：**
        *   **种类 (lower_type)：** 长裤、短裤、裙子等。
        *   **材质 (lower_material)：** 棉、牛仔、涤纶等（根据视觉判断）。
        *   **覆盖面积 (lower_coverage)：** 估计裤子覆盖腿部的比例（例如，覆盖整个腿部，覆盖膝盖以上等）。
        *  **细节 (lower_details):** 是否宽松、紧身等
    *   **补充分析：**
        *   **帽子 (hat)：** 是否佩戴帽子，帽子种类（棒球帽、毛线帽等），材质。
        *   **围巾 (scarf)：** 是否佩戴围巾，围巾材质。
        *   **椅子接触面积 (chair_contact)：** 估计人员与椅子接触的身体面积，这会影响服装的热阻。
        *   **其他 (additional_info)：** 如果有其他影响服装热阻的因素，例如，是否穿着马甲，是否穿着袜子等，也会进行描述。
3.  **遮挡处理：** 如果有衣物被遮挡，会根据可见部分进行合理估计。
4.  **JSON 输出：** 将分析结果整理成 JSON 格式输出。

## Result
用以下的json格式输出:
```json
{
  "analyze": "{{your analysis}}",
  "clothes": {
    "upper": {
      "type": "{{衬衫}}",
      "material": "{{棉}}",
      "coverage": "{{覆盖上半身，长袖}}",
      "details": "{{有领子，袖子扣着}}"
    },
    "lower": {
      "type": "{{长裤}}",
      "material": "{{牛仔}}",
      "coverage": "{{覆盖整个腿部}}",
      "details": "{{宽松}}"
    },
    "extras": {
      "hat": "{{无}}",
      "scarf": "{{无}}",
      "chair": "{{背部和臀部}}",
      "other": "{{穿着袜子}}"
    }
  }
}
```

""".strip()
    return prompt

## ========== @ ai.analyze_video_302.py ==========
def get_analyze_video_prompt():
    prompt = """
## 背景
您是一位专门研究室内人体动作与热舒适度关系的专家。您擅长分析短视频中的人体动作，并判断这些动作是否与冷热不舒适状态有关。您需要分析一段5秒钟的室内视频。

## 目标
1. 识别并描述视频中的主要人体动作和状态，无论是否与热舒适状态有关。
2. 判断这些动作是否与热舒适状态(thermal comfort)有关。
3. 提供简要解释，说明为什么这些动作可能表示热舒适状态或仅是正常活动。

## 注意事项
1. 某些动作可能只是正常的活动，不一定与冷热不舒适有关。在做出判断时，请考虑动作的目的、频率和强度。
2. 需要识别的动作不仅限于**[用手扇风、用手抖衣服鼓风、用手擦汗、双手环抱、给手吹风、搓手取暖等]**，还可能包括其他表示冷热不舒适的动作。
3. 请留意视频中的其他环境因素，如可见的空调设备、开着的窗户等，这些可能会帮助您更准确地判断热舒适状态。

## 响应
请使用以下 JSON 格式提供您的分析：
```json
{
  "action": "简要描述人员的一般动作和状态",
  "tc_related": "是/否",
  "explanation": "解释为什么这些动作可能表示冷热不舒适或仅是正常活动"
}
```

## 示例
```json
{
  "action": "一个人坐在椅子上，快速地用手扇风",
  "tc_related": "是",
  "explanation": "虽然坐着是一种常见的姿势，但快速用手扇风通常表示感到热不舒适。这是一种常见的降温行为，暗示了可能存在热不舒适状态。"
}
```
""".strip()
    return prompt