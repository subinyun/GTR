#!/usr/bin/env python3
"""Chinese legal element axis schema for CAIL2018 GTR v2 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def article(num: str | int) -> str:
    return f"中华人民共和国刑法 第{num}条"


AXIS_SCHEMA: List[Dict[str, object]] = [
    {
        "axis_id": "personal_violence_injury",
        "chinese_name": "人身暴力/伤害结果",
        "english_name": "personal_violence_injury",
        "description": "殴打、伤害、杀害等侵犯人身安全并造成伤亡结果的事实要素",
        "positive_keywords": ["故意伤害", "殴打", "拳头", "持刀", "砍伤", "打伤", "轻伤", "重伤", "杀害", "捅伤"],
        "negative_keywords": ["未造成伤害", "没有受伤", "未受伤"],
        "related_statutes": [article(232), article(233), article(234), article(235)],
        "hard_negative_statutes": [article(234), article(232), article(233)],
    },
    {
        "axis_id": "violence_coercion_robbery",
        "chinese_name": "暴力胁迫取财",
        "english_name": "violence_coercion_robbery",
        "description": "以暴力、胁迫或当场强制方式取得财物的事实要素",
        "positive_keywords": ["抢劫", "抢夺", "暴力", "胁迫", "威胁", "持刀抢", "当场劫取", "强行夺取"],
        "negative_keywords": ["未使用暴力", "没有胁迫"],
        "related_statutes": [article(263), article(267), article(269)],
        "hard_negative_statutes": [article(264), article(266), article(274)],
    },
    {
        "axis_id": "property_secret_or_illegal_taking",
        "chinese_name": "非法占有/秘密窃取财物",
        "english_name": "property_secret_or_illegal_taking",
        "description": "秘密窃取、侵占、职务便利占有或毁坏财物等财产犯罪事实要素",
        "positive_keywords": ["盗窃", "秘密窃取", "窃取", "扒窃", "入户盗窃", "侵占", "职务侵占", "挪用", "非法占有", "毁坏"],
        "negative_keywords": ["未取得财物", "没有非法占有"],
        "related_statutes": [article(264), article(270), article(271), article(272), article(275)],
        "hard_negative_statutes": [article(263), article(266), article(274)],
    },
    {
        "axis_id": "deception_fraud",
        "chinese_name": "诈骗/虚假意思表示",
        "english_name": "deception_fraud",
        "description": "虚构事实、隐瞒真相、骗取财物或金融利益的事实要素",
        "positive_keywords": ["诈骗", "骗取", "虚构", "隐瞒真相", "虚假", "冒充", "信用卡", "合同诈骗", "非法占有为目的"],
        "negative_keywords": ["未骗取", "没有虚构事实"],
        "related_statutes": [article(192), article(193), article(196), article(224), article(266)],
        "hard_negative_statutes": [article(264), article(270), article(271)],
    },
    {
        "axis_id": "public_safety_danger",
        "chinese_name": "危害公共安全",
        "english_name": "public_safety_danger",
        "description": "放火、爆炸、危险方法、重大事故等危及不特定多数人安全的事实要素",
        "positive_keywords": ["放火", "爆炸", "失火", "危险方法", "公共安全", "重大事故", "安全事故", "火灾", "爆炸物"],
        "negative_keywords": ["未危及公共安全"],
        "related_statutes": [article(114), article(115), article(125), article(134), article(136)],
        "hard_negative_statutes": [article(133), article(232), article(234)],
    },
    {
        "axis_id": "traffic_driving",
        "chinese_name": "交通驾驶/道路安全",
        "english_name": "traffic_driving",
        "description": "机动车驾驶、交通肇事、危险驾驶、酒驾等道路交通事实要素",
        "positive_keywords": ["交通肇事", "危险驾驶", "醉酒驾驶", "酒后驾驶", "血液酒精", "道路交通事故"],
        "negative_keywords": ["未驾驶", "非机动车"],
        "related_statutes": [article(133)],
        "hard_negative_statutes": [article(114), article(115), article(134)],
    },
    {
        "axis_id": "drugs",
        "chinese_name": "毒品犯罪",
        "english_name": "drugs",
        "description": "走私、贩卖、运输、制造、持有、容留或引诱吸食毒品的事实要素",
        "positive_keywords": ["毒品", "甲基苯丙胺", "冰毒", "海洛因", "贩卖", "运输", "制造", "吸毒", "容留", "麻古"],
        "negative_keywords": ["未查获毒品"],
        "related_statutes": [article(347), article(348), article(353), article(354), article(356), article(357)],
        "hard_negative_statutes": [article(347), article(354), article(348)],
    },
    {
        "axis_id": "official_corruption_bribery",
        "chinese_name": "贪污贿赂/职务廉洁",
        "english_name": "official_corruption_bribery",
        "description": "国家工作人员职务便利、贪污、受贿、行贿、滥用职权等职务犯罪事实要素",
        "positive_keywords": ["国家工作人员", "职务便利", "贪污", "受贿", "行贿", "回扣", "滥用职权", "玩忽职守", "公款"],
        "negative_keywords": ["非国家工作人员", "未利用职务便利"],
        "related_statutes": [article(382), article(383), article(385), article(386), article(389), article(390), article(397)],
        "hard_negative_statutes": [article(271), article(272), article(266)],
    },
    {
        "axis_id": "documents_seals_identity",
        "chinese_name": "证件印章/身份文书",
        "english_name": "documents_seals_identity",
        "description": "伪造、变造、买卖国家机关证件、印章、身份证件等文书身份事实要素",
        "positive_keywords": ["伪造", "变造", "买卖", "证件", "印章", "身份证", "国家机关", "公文", "驾驶证"],
        "negative_keywords": ["真实证件", "未伪造"],
        "related_statutes": [article(280)],
        "hard_negative_statutes": [article(205), article(266)],
    },
    {
        "axis_id": "tax_invoice_economic_order",
        "chinese_name": "税票/市场经济秩序",
        "english_name": "tax_invoice_economic_order",
        "description": "虚开增值税专用发票、非法经营、扰乱市场经济秩序等事实要素",
        "positive_keywords": ["虚开", "增值税", "发票", "骗取出口退税", "抵扣税款", "非法经营", "未经许可", "经营"],
        "negative_keywords": ["未开具发票"],
        "related_statutes": [article(201), article(205), article(225)],
        "hard_negative_statutes": [article(266), article(224)],
    },
    {
        "axis_id": "food_drug_product_safety",
        "chinese_name": "食品药品/产品安全",
        "english_name": "food_drug_product_safety",
        "description": "生产、销售假药、劣药、有毒有害食品或不符合安全标准食品的事实要素",
        "positive_keywords": ["假药", "劣药", "有毒", "有害食品", "食品", "药品", "不符合安全标准", "生产", "销售"],
        "negative_keywords": ["符合安全标准"],
        "related_statutes": [article(140), article(141), article(142), article(143), article(144)],
        "hard_negative_statutes": [article(225), article(266)],
    },
    {
        "axis_id": "environment_resources_forestry",
        "chinese_name": "环境资源/林地林木",
        "english_name": "environment_resources_forestry",
        "description": "污染环境、非法占用农用地、盗伐滥伐林木等环境资源事实要素",
        "positive_keywords": ["污染环境", "排放", "废水", "废物", "农用地", "林地", "林木", "盗伐", "滥伐", "采伐"],
        "negative_keywords": ["未污染", "合法采伐"],
        "related_statutes": [article(338), article(342), article(344), article(345)],
        "hard_negative_statutes": [article(225), article(264)],
    },
    {
        "axis_id": "guns_weapons_explosives",
        "chinese_name": "枪支弹药/危险物品",
        "english_name": "guns_weapons_explosives",
        "description": "非法制造、买卖、运输、持有枪支弹药爆炸物等危险物品事实要素",
        "positive_keywords": ["枪支", "弹药", "爆炸物", "火药", "非法持有", "私藏", "买卖枪支", "猎枪"],
        "negative_keywords": ["未持有枪支"],
        "related_statutes": [article(125), article(128), article(130)],
        "hard_negative_statutes": [article(114), article(115)],
    },
    {
        "axis_id": "sexual_offense",
        "chinese_name": "性侵/强制猥亵",
        "english_name": "sexual_offense",
        "description": "强奸、强制猥亵、侮辱妇女、儿童性侵等性犯罪事实要素",
        "positive_keywords": ["强奸", "猥亵", "性关系", "奸淫", "妇女", "幼女", "被害女", "违背妇女意志"],
        "negative_keywords": ["自愿发生性关系"],
        "related_statutes": [article(236), article(237)],
        "hard_negative_statutes": [article(358), article(359)],
    },
    {
        "axis_id": "prostitution_exploitation",
        "chinese_name": "卖淫组织/容留介绍",
        "english_name": "prostitution_exploitation",
        "description": "组织、强迫、引诱、容留、介绍卖淫等性交易组织事实要素",
        "positive_keywords": ["卖淫", "嫖娼", "组织卖淫", "强迫卖淫", "容留", "介绍卖淫", "引诱"],
        "negative_keywords": ["未组织卖淫"],
        "related_statutes": [article(358), article(359)],
        "hard_negative_statutes": [article(236), article(237)],
    },
    {
        "axis_id": "public_order_gambling_disruption",
        "chinese_name": "社会秩序/赌博滋事",
        "english_name": "public_order_gambling_disruption",
        "description": "寻衅滋事、聚众斗殴、赌博、扰乱社会管理秩序等事实要素",
        "positive_keywords": ["寻衅滋事", "聚众斗殴", "赌博", "赌场", "扰乱", "起哄闹事", "随意殴打", "追逐拦截"],
        "negative_keywords": ["未扰乱秩序"],
        "related_statutes": [article(292), article(293), article(303)],
        "hard_negative_statutes": [article(234), article(264)],
    },
    {
        "axis_id": "obstruction_official_judicial",
        "chinese_name": "妨害公务/司法执行",
        "english_name": "obstruction_official_judicial",
        "description": "阻碍国家机关工作人员依法执行职务、拒不执行判决裁定等事实要素",
        "positive_keywords": ["妨害公务", "阻碍执行职务", "抗拒执法", "暴力抗法", "拒不执行判决", "拒不执行裁定"],
        "negative_keywords": ["未阻碍执法"],
        "related_statutes": [article(277), article(313)],
        "hard_negative_statutes": [article(293), article(397)],
    },
    {
        "axis_id": "conceal_harbor_criminal_proceeds",
        "chinese_name": "窝藏包庇/掩饰隐瞒犯罪所得",
        "english_name": "conceal_harbor_criminal_proceeds",
        "description": "窝藏、包庇、掩饰、隐瞒犯罪所得及其收益等事后帮助事实要素",
        "positive_keywords": ["窝藏", "包庇", "掩饰", "隐瞒", "犯罪所得", "赃物", "销赃", "明知"],
        "negative_keywords": ["不明知"],
        "related_statutes": [article(310), article(312)],
        "hard_negative_statutes": [article(264), article(266)],
    },
    {
        "axis_id": "labor_employment_obligation",
        "chinese_name": "劳动报酬/用工义务",
        "english_name": "labor_employment_obligation",
        "description": "拒不支付劳动报酬、逃避支付工资等劳动权益事实要素",
        "positive_keywords": ["劳动报酬", "工资", "拖欠", "拒不支付", "工人", "逃匿"],
        "negative_keywords": ["已支付工资"],
        "related_statutes": [article(276)],
        "hard_negative_statutes": [article(271), article(272)],
    },
    {
        "axis_id": "joint_group_crime",
        "chinese_name": "共同犯罪/聚众纠集",
        "english_name": "joint_group_crime",
        "description": "共同犯罪、纠集多人、聚众、团伙分工等多人参与事实要素",
        "positive_keywords": ["共同", "伙同", "纠集", "聚众", "团伙", "分工", "同案", "多人"],
        "negative_keywords": ["单独作案"],
        "related_statutes": [article(25), article(26), article(27), article(292), article(293)],
        "hard_negative_statutes": [],
    },
]


def get_axis_schema() -> List[Dict[str, object]]:
    """Return a copy of the CAIL axis schema."""
    return [dict(item) for item in AXIS_SCHEMA]


def get_axis_ids() -> List[str]:
    """Return CAIL axis ids in canonical order."""
    return [str(item["axis_id"]) for item in AXIS_SCHEMA]


def get_statute_to_axes_map() -> Dict[str, List[str]]:
    """Build statute -> related CAIL axis ids from the schema."""
    mapping: Dict[str, List[str]] = {}
    for item in AXIS_SCHEMA:
        axis_id = str(item["axis_id"])
        for statute in item.get("related_statutes", []):
            mapping.setdefault(str(statute), [])
            if axis_id not in mapping[str(statute)]:
                mapping[str(statute)].append(axis_id)
    return mapping


def save_axis_schema(path: str | Path) -> None:
    """Save the schema as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(get_axis_schema(), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    save_axis_schema(Path("output/cail2018_gtr_v2_only/axis_schema.json"))
