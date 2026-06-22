#!/usr/bin/env python3
"""Legal element axis schema for Full GTR experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


AXIS_SCHEMA: List[Dict[str, object]] = [
    {
        "axis_id": "violence_or_threat",
        "korean_name": "폭행/협박성",
        "english_name": "violence_or_threat",
        "description": "폭행, 협박, 위력 행사 등 유형력 또는 해악 고지가 있었는지 여부",
        "positive_keywords": ["폭행", "협박", "위협", "때리", "밀치", "주먹", "발로", "위력", "강제로"],
        "negative_keywords": ["폭행하지", "협박하지", "유형력을 행사하지"],
        "related_statutes": ["형법 제136조 제1항", "형법 제257조 제1항", "형법 제260조 제1항", "형법 제283조 제1항", "형법 제298조", "형법 제333조"],
        "hard_negative_statutes": ["형법 제257조 제1항", "형법 제260조 제1항", "형법 제136조 제1항"],
    },
    {
        "axis_id": "injury_result",
        "korean_name": "상해결과성",
        "english_name": "injury_result",
        "description": "피해자에게 치료가 필요한 신체적 상해 결과가 발생했는지 여부",
        "positive_keywords": ["상해", "치료", "진단", "골절", "찰과상", "타박상", "염좌", "상처", "전치"],
        "negative_keywords": ["상해를 입지", "상해가 발생하지", "치료를 받지", "다치지"],
        "related_statutes": ["형법 제257조 제1항"],
        "hard_negative_statutes": ["형법 제260조 제1항"],
    },
    {
        "axis_id": "official_victim",
        "korean_name": "공무원성",
        "english_name": "official_victim",
        "description": "피해자 또는 상대방이 공무원, 경찰관, 소방관, 공무수행자 등인지 여부",
        "positive_keywords": ["경찰관", "공무원", "소방관", "단속", "공무수행", "공무집행", "공무소"],
        "negative_keywords": ["일반인", "민간인", "공무원이 아닌"],
        "related_statutes": ["형법 제136조 제1항"],
        "hard_negative_statutes": ["형법 제260조 제1항"],
    },
    {
        "axis_id": "duty_execution",
        "korean_name": "직무집행성",
        "english_name": "duty_execution",
        "description": "상대방이 단속, 체포, 신고 처리 등 직무를 집행하는 중이었는지 여부",
        "positive_keywords": ["직무집행", "직무 수행", "단속 중", "체포하려", "신고를 받고", "출동", "공무집행"],
        "negative_keywords": ["직무와 무관", "근무 중이 아닌", "사적으로"],
        "related_statutes": ["형법 제136조 제1항"],
        "hard_negative_statutes": ["형법 제260조 제1항"],
    },
    {
        "axis_id": "lawful_duty",
        "korean_name": "적법직무성",
        "english_name": "lawful_duty",
        "description": "공무원의 직무집행이 법령상 요건과 절차를 갖춘 적법한 집행인지 여부",
        "positive_keywords": ["적법", "정당한 직무", "법령에 따라", "현행범", "영장"],
        "negative_keywords": ["위법", "적법하지", "권한 없이", "절차를 위반"],
        "related_statutes": ["형법 제136조 제1항"],
        "hard_negative_statutes": ["형법 제260조 제1항"],
    },
    {
        "axis_id": "property_taken",
        "korean_name": "재물취득/탈취",
        "english_name": "property_taken",
        "description": "재물 또는 재산상 이익을 취득, 절취, 강취, 편취했는지 여부",
        "positive_keywords": ["절취", "훔쳐", "가져가", "강취", "편취", "교부받", "재물", "재산상 이익"],
        "negative_keywords": ["취득하지", "가져가지", "재물을 반환"],
        "related_statutes": ["형법 제329조", "형법 제333조", "형법 제347조 제1항", "형법 제355조 제1항"],
        "hard_negative_statutes": ["형법 제329조", "형법 제333조", "형법 제347조 제1항", "형법 제355조 제1항"],
    },
    {
        "axis_id": "unlawful_gain_intent",
        "korean_name": "불법영득의사",
        "english_name": "unlawful_gain_intent",
        "description": "권리자를 배제하고 자기 또는 제3자의 소유물처럼 이용하려는 의사가 있었는지 여부",
        "positive_keywords": ["불법영득", "영득", "임의로 사용", "자신의 소유", "처분"],
        "negative_keywords": ["반환할 의사", "일시 사용", "영득의사가 없"],
        "related_statutes": ["형법 제329조", "형법 제333조", "형법 제355조 제1항"],
        "hard_negative_statutes": ["형법 제329조", "형법 제355조 제1항"],
    },
    {
        "axis_id": "deception",
        "korean_name": "기망행위",
        "english_name": "deception",
        "description": "거짓말, 허위 고지, 착오 유발 등 기망행위가 있었는지 여부",
        "positive_keywords": ["기망", "거짓말", "허위", "속여", "착오", "편취", "사칭"],
        "negative_keywords": ["기망하지", "사실대로", "착오가 없"],
        "related_statutes": ["형법 제347조 제1항"],
        "hard_negative_statutes": ["형법 제355조 제1항", "형법 제355조 제2항"],
    },
    {
        "axis_id": "entrustment_or_custody",
        "korean_name": "위탁/보관관계",
        "english_name": "entrustment_or_custody",
        "description": "타인의 재물을 보관하거나 위탁받은 지위에서 처분했는지 여부",
        "positive_keywords": ["보관", "위탁", "관리", "맡긴", "임무에 위배", "횡령", "배임"],
        "negative_keywords": ["보관자가 아닌", "위탁받지", "관리하지"],
        "related_statutes": ["형법 제355조 제1항", "형법 제355조 제2항"],
        "hard_negative_statutes": ["형법 제347조 제1항", "형법 제329조"],
    },
    {
        "axis_id": "sexual_act",
        "korean_name": "추행/성적 행위",
        "english_name": "sexual_act",
        "description": "추행, 성적 접촉, 간음 등 성적 행위가 있었는지 여부",
        "positive_keywords": ["추행", "성추행", "간음", "성관계", "가슴", "엉덩이", "입맞춤", "성적"],
        "negative_keywords": ["추행하지", "성적 의도 없이", "접촉하지"],
        "related_statutes": ["형법 제298조", "형법 제299조"],
        "hard_negative_statutes": ["형법 제298조", "형법 제299조"],
    },
    {
        "axis_id": "inability_to_resist",
        "korean_name": "항거불능/심신상실",
        "english_name": "inability_to_resist",
        "description": "피해자가 술, 잠, 심신상실 등으로 항거불능 또는 항거곤란 상태였는지 여부",
        "positive_keywords": ["항거불능", "심신상실", "잠이 든", "술에 취해", "의식이 없", "만취"],
        "negative_keywords": ["항거불능이 아닌", "의식이 있", "저항할 수 있"],
        "related_statutes": ["형법 제299조"],
        "hard_negative_statutes": ["형법 제298조"],
    },
    {
        "axis_id": "joint_principal",
        "korean_name": "공동정범/공모성",
        "english_name": "joint_principal",
        "description": "2인 이상이 공모하거나 공동으로 실행했는지 여부",
        "positive_keywords": ["공모", "공동", "함께", "합동", "공동하여", "일행"],
        "negative_keywords": ["단독", "공모하지", "혼자"],
        "related_statutes": ["형법 제30조", "형법 제331조 제2항"],
        "hard_negative_statutes": [],
    },
    {
        "axis_id": "special_weapon_or_group",
        "korean_name": "위험한 물건/단체 또는 다중 위력",
        "english_name": "special_weapon_or_group",
        "description": "흉기, 위험한 물건, 단체 또는 다중의 위력을 이용했는지 여부",
        "positive_keywords": ["흉기", "위험한 물건", "칼", "망치", "병", "단체", "다중", "특수"],
        "negative_keywords": ["위험한 물건 없이", "흉기를 사용하지"],
        "related_statutes": ["형법 제261조", "형법 제284조", "형법 제331조 제2항"],
        "hard_negative_statutes": ["형법 제260조 제1항", "형법 제283조 제1항"],
    },
    {
        "axis_id": "traffic_driving",
        "korean_name": "운전행위",
        "english_name": "traffic_driving",
        "description": "자동차, 원동기장치자전거 등 교통수단 운전행위가 있었는지 여부",
        "positive_keywords": ["운전", "자동차", "차량", "승용차", "화물차", "도로", "교통사고", "무면허"],
        "negative_keywords": ["운전하지", "차량을 운행하지"],
        "related_statutes": ["도로교통법 제43조", "도로교통법 제44조 제1항", "교통사고처리 특례법 제3조 제1항"],
        "hard_negative_statutes": ["도로교통법 제43조", "도로교통법 제44조 제1항"],
    },
    {
        "axis_id": "intoxication",
        "korean_name": "음주/약물 영향",
        "english_name": "intoxication",
        "description": "음주, 약물 또는 이에 준하는 영향 상태였는지 여부",
        "positive_keywords": ["음주", "술에 취", "혈중알코올", "만취", "약물", "마약"],
        "negative_keywords": ["술에 취하지", "음주하지", "혈중알코올농도 0"],
        "related_statutes": ["도로교통법 제44조 제1항", "형법 제299조"],
        "hard_negative_statutes": ["도로교통법 제43조"],
    },
]


def get_axis_schema() -> List[Dict[str, object]]:
    """Return a copy of the axis schema."""
    return [dict(item) for item in AXIS_SCHEMA]


def get_axis_ids() -> List[str]:
    """Return axis ids in canonical order."""
    return [str(item["axis_id"]) for item in AXIS_SCHEMA]


def get_statute_to_axes_map() -> Dict[str, List[str]]:
    """Build statute -> related axis ids from the schema."""
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
    save_axis_schema(Path("output/full_gtr/axis_schema.json"))
