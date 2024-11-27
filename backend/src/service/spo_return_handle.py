import json
import logging

from dto.spo_dto import EntityData, SpoItemData

logger = logging.getLogger(__name__)


def common_return_handle(json_text: str, **kwargs) -> list[SpoItemData]:
    json_dict = json.loads(json_text)
    spos = []
    if not isinstance(json_dict,dict):
        return []
    for item in json_dict.get("spo",[]):
        try:
            subject = EntityData(**item["subject"])
            object = EntityData(**item["object"])
            data = SpoItemData(subject=subject, relation=item["relation"], object=object)
            spos.append(data)
        except Exception as e:
            logger.error(f"json_text:{json_text}")
            logger.error(f"error:{e}")
    return spos


def tour_return_handle(json_text: str, root_obj: str) -> list[SpoItemData]:
    try:
        # 修正后的JSON
        if json_text.endswith('"'):
            json_text = json_text[:-1]
        json_text = json_text.replace("‘","")
        try:
            json_dict = json.loads(json_text)
        except Exception as e:
            logger.error(f"json_text:{json_text}")
            return []
        spos = []
        obj_to_sub = {}
        if not isinstance(json_dict,dict):
            return []
        for item in json_dict.get("spo",[]):
            if not item.get("subject"):
                continue
            subject = EntityData(**item["subject"])
            relation = item.get("relation", "包含")
            if not item.get("object") or not item["object"].get("name"):
                object = subject
                subject = EntityData(type="Tour", name=root_obj)
                if not item.get("object"):
                    relation = "包含"
            else:
                object = EntityData(**item["object"])
                obj_to_sub[object.name] = subject
            data = SpoItemData(subject=subject, relation=relation, object=object)
            spos.append(data)
        spo_res = []
        for spo in spos:
            if not obj_to_sub.get(spo.subject.name) and spo.subject.name != root_obj:
                obj_to_sub[spo.subject.name] = EntityData(type="Tour", name=root_obj)
                spo_res.append(
                    SpoItemData(subject=EntityData(type="Tour", name=root_obj), relation="包含", object=spo.subject))
            spo_res.append(spo)
        return spo_res
    except Exception as e:
        logger.error(f"json_text:{json_text}")
        logger.error(f"error:{e}")
        return []

def museum_return_handle(json_text: str, root_obj: str) -> list[SpoItemData]:
    try:
        # 修正后的JSON
        if json_text.endswith('"'):
            json_text = json_text[:-1]
        json_text = json_text.replace("‘","")
        try:
            json_dict = json.loads(json_text)
        except Exception as e:
            logger.error(f"json_text:{json_text}")
            return []
        spos = []
        obj_to_sub = {}
        if not isinstance(json_dict,dict):
            return []
        for item in json_dict.get("spo",[]):
            if not item.get("subject"):
                continue
            subject = EntityData(**item["subject"])
            relation = item.get("relation", "包含")
            if not item.get("object") or not item["object"].get("name"):
                object = subject
                subject = EntityData(type="CulturalRelics", name=root_obj)
                if not item.get("object"):
                    relation = "包含"
            else:
                object = EntityData(**item["object"])
                obj_to_sub[object.name] = subject
            data = SpoItemData(subject=subject, relation=relation, object=object)
            spos.append(data)
        spo_res = []
        for spo in spos:
            if not obj_to_sub.get(spo.subject.name) and spo.subject.name != root_obj:
                obj_to_sub[spo.subject.name] = EntityData(type="CulturalRelics", name=root_obj)
                spo_res.append(
                    SpoItemData(subject=EntityData(type="CulturalRelics", name=root_obj), relation="包含", object=spo.subject))
            spo_res.append(spo)
        return spo_res
    except Exception as e:
        logger.error(f"json_text:{json_text}")
        logger.error(f"error:{e}")
        return []