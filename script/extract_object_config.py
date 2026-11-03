import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import threading


# ============================================================
# 0. 直接在这里改参数，不使用 argparse
# ============================================================

# 输入：一个 JSONL 文件，里面每行是一条视频记录
INPUT_JSONL = "prompts/interactive_benchmark.jsonl"

# 输出：所有 object_config 汇总成一个 JSONL
OUTPUT_JSONL = "third_party/ReferDINO/output/referdino_soci/object_configs.jsonl"

# 输出：每个视频单独保存一个 object_config.json，方便 ReferDINO 逐个读取
OUTPUT_CONFIG_DIR = "third_party/ReferDINO/output/referdino_soci/object_configs"

# 失败日志
FAILED_JSONL = "third_party/ReferDINO/output/referdino_soci/failed_object_configs.jsonl"

# JSONL 中 prompt 列表的 key
PROMPTS_KEY = "prompts"

# JSONL 中视频 id 的 key；如果没有，会自动用 line_000000
VIDEO_ID_KEY = "video_id"

# JSONL 中视频路径的 key；没有也没关系
VIDEO_PATH_KEY = "video_path"

# 每段 prompt 持续时间。你的例子是每 10 秒切换一次 prompt
PROMPT_DURATION = 10.0

# LLM 参数
MODEL = "deepseek-flash"

# 如果你用 OpenAI 官方接口：
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.deepseek.com"

# 如果你用 OpenAI-compatible 接口，可以直接写死，例如：
# API_KEY = "sk-xxx"
# BASE_URL = "https://api.xxx.com/v1"

# 抽取对象数量上限
MAX_OBJECTS = 5

# 默认偏好至少出现 2 个 prompt 段的稳定对象
MIN_ACTIVE_SEGMENTS = 2

# 是否允许只出现 1 个 prompt 段的对象
# 如果你的视频里有重要对象短暂出现，可以设 True
ALLOW_SINGLE_SEGMENT_OBJECT = True

# 断点续跑：如果 OUTPUT_JSONL 中已经有该 video_id，就跳过
RESUME = True

# 如果单独 object_config.json 已存在，是否覆盖
OVERWRITE_INDIVIDUAL_CONFIG = True

# LLM 调用重试
MAX_RETRIES = 3
RETRY_SLEEP = 3.0

# 在文件顶部（约第 10 行后）增加导入
from concurrent.futures import ThreadPoolExecutor, as_completed

# 在参数区末尾（约第 80 行后）增加
CONCURRENCY = 10  # 并发线程数，可根据 API 限频调整，建议 3~10

# 使用 JSON Schema 结构化输出。
# 如果你的 OpenAI-compatible 服务不支持 json_schema，可以改成 "json_object"
# 可选值: "json_schema", "json_object"
RESPONSE_FORMAT_MODE = "json_object"

# temperature 建议 0，保证批量处理稳定
TEMPERATURE = 0.0


# ============================================================
# 1. Prompt 设计
# ============================================================

SYSTEM_PROMPT = r"""
You extract stable, visually segmentable foreground INSTANCES from a sequence of
video-generation prompt segments, so that a referring-expression segmentation model
can ground each instance per segment.

An INSTANCE is a concrete foreground entity that appears in at least one segment:
character, animal, creature, vehicle, handheld tool, major movable prop.

HARD RULES
1. One instance = one entry. Merge all coreferent mentions across segments
   (definite/indefinite/pronoun/abbreviated forms) into a single entry.
2. "text" must be a SHORT noun phrase (2-8 words) that uniquely identifies THIS
   instance within the whole video. It MUST contain at least one STABLE visual
   attribute (color / material / shape / clothing / size / species).
   It MUST NOT contain actions, locations, camera terms, emotions, time words.
3. "active_prompt_indices": list EVERY 0-based segment index where the instance
   is present. Include a segment if the instance is explicitly mentioned OR
   clearly implied (pronoun reference, scene continuity, introduced and never left).
4. Order "objects" by descending salience.
5. Extract only what is visually groundable. Do NOT extract:
   backgrounds, rooms, walls, sky, weather, lighting, shadows, mood,
   camera motion, abstract events, emotions, vague plurals
   ("objects", "things", "people" as a group).
   Effects (light, smoke, fire, sparks, particles) are extracted ONLY if they
   are the primary interaction target across >= 2 segments AND have a concrete
   visual form; assign role="effect".
6. If nothing is groundable, return an empty objects list.
7. Output JSON only. No prose.

SALIENCE SCALE (1-5)
5 = main character(s) / primary subject
4 = secondary character OR primary interactive effect
3 = tool the main character actively holds/uses
2 = short-lived but clearly visible prop
1 = barely visible / ambiguous / borderline

ROLE ENUM
"main_character" | "secondary_character" | "tool" | "prop" | "effect"

EXAMPLE A (multi-character merge)
Segments:
[0] A woman in a red dress walks into a cafe. She carries a black umbrella.
[1] The woman sits down. The umbrella leans against the chair.
[2] A man in a blue suit joins her. He holds a laptop.
Output:
{"objects":[
  {"name":"woman","text":"woman in a red dress","role":"main_character","salience":5,"active_prompt_indices":[0,1,2]},
  {"name":"man","text":"man in a blue suit","role":"main_character","salience":5,"active_prompt_indices":[2]},
  {"name":"umbrella","text":"black umbrella","role":"tool","salience":3,"active_prompt_indices":[0,1]},
  {"name":"laptop","text":"laptop","role":"prop","salience":2,"active_prompt_indices":[2]}
]}

EXAMPLE B (effect as primary target)
Segments:
[0] A scientist in a lab coat pours liquid into a beaker. Blue smoke rises.
[1] The blue smoke forms a swirling sphere above the beaker.
[2] The scientist watches the swirling sphere expand.
Output:
{"objects":[
  {"name":"scientist","text":"scientist in a lab coat","role":"main_character","salience":5,"active_prompt_indices":[0,1,2]},
  {"name":"smoke_sphere","text":"swirling blue smoke sphere","role":"effect","salience":4,"active_prompt_indices":[1,2]}
]}

EXAMPLE C (nothing groundable)
Segments:
[0] A wide shot of a sunset over the ocean. Waves gently roll.
Output:
{"objects":[]}
"""


def build_user_prompt(
    prompts: List[str],
    prompt_duration: float,
    max_objects: int,
    min_active_segments: int,
    allow_single_segment_object: bool,
) -> str:
    numbered_prompts = []
    for i, p in enumerate(prompts):
        numbered_prompts.append(f"[{i}] {p}")

    allow_text = (
        "Objects that appear in only one segment MAY still be included if visually important."
        if allow_single_segment_object
        else f"Only include objects that appear in at least {min_active_segments} prompt segments."
    )

    return f"""
We have a video generated from the following prompt segments.
Each prompt segment lasts {prompt_duration} seconds.

Extract stable foreground object instances for video object segmentation.

Constraints:
- Maximum number of objects: {max_objects}
- {allow_text}
- Use 0-based prompt segment indices.
- The output will be used as ReferDINO object text.
- Do not include scene/background/effects/actions as objects unless the rules above allow it.
- For every object, decide role and salience per the system rules.

Prompt segments:
{chr(10).join(numbered_prompts)}

Return JSON with exactly this structure:
{{
  "prompt_duration": {prompt_duration},
  "prompt_count": {len(prompts)},
  "objects": [
    {{
      "name": "short_object_name",
      "text": "short visual referring expression",
      "role": "main_character | secondary_character | tool | prop | effect",
      "salience": 1,
      "active_prompt_indices": [0, 1, 2]
    }}
  ]
}}
"""


# ============================================================
# 2. JSON Schema
# ============================================================

def get_object_config_schema() -> Dict[str, Any]:
    return {
        "name": "referdino_object_config",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompt_duration": {"type": "number"},
                "prompt_count": {"type": "integer"},
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "text": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "main_character",
                                    "secondary_character",
                                    "tool",
                                    "prop",
                                    "effect"
                                ]
                            },
                            "salience": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "active_prompt_indices": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": [
                            "name",
                            "text",
                            "role",
                            "salience",
                            "active_prompt_indices"
                        ]
                    }
                }
            },
            "required": ["prompt_duration", "prompt_count", "objects"]
        }
    }


# ============================================================
# 3. 工具函数
# ============================================================

def safe_video_id(raw_id: Any, line_idx: int) -> str:
    if raw_id is None or str(raw_id).strip() == "":
        s = f"line_{line_idx:06d}"
    else:
        s = str(raw_id).strip()

    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = s.strip("_")

    if not s:
        s = f"line_{line_idx:06d}"

    return s

ROLE_ORDER = {
    "main_character": 0,
    "secondary_character": 1,
    "tool": 2,
    "prop": 3,
    "effect": 4,
}
def sanitize_object_name(name: str, fallback: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_ -]+", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    name = name.strip("_")
    return name if name else fallback


def clean_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    if not text:
        return text
    return text[0].upper() + text[1:]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                yield line_idx, None, f"invalid_json: {repr(e)}"
                continue

            yield line_idx, record, None


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_done_video_ids(output_jsonl: str) -> set:
    done = set()

    if not os.path.exists(output_jsonl):
        return done

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            video_id = item.get("video_id")
            if video_id:
                done.add(str(video_id))

    return done


def extract_prompts(record: Dict[str, Any], prompts_key: str) -> List[str]:
    if prompts_key not in record:
        raise ValueError(f"Missing key: {prompts_key}")

    prompts = record[prompts_key]

    # 有些数据可能把 prompts 存成 JSON 字符串
    if isinstance(prompts, str):
        s = prompts.strip()

        # 尝试解析成 list
        if s.startswith("[") and s.endswith("]"):
            try:
                prompts = json.loads(s)
            except Exception:
                prompts = [s]
        else:
            prompts = [s]

    if not isinstance(prompts, list):
        raise ValueError(f"{prompts_key} must be a list or string.")

    prompts = [str(x).strip() for x in prompts if str(x).strip()]

    if len(prompts) == 0:
        raise ValueError("Empty prompts.")

    return prompts


def validate_and_normalize_config(
    raw_cfg: Dict[str, Any],
    prompt_count: int,
    prompt_duration: float,
    max_objects: int,
    min_active_segments: int,
    allow_single_segment_object: bool,
) -> Dict[str, Any]:
    if not isinstance(raw_cfg, dict):
        raise ValueError("LLM output is not a dict.")

    raw_objects = raw_cfg.get("objects", [])
    if not isinstance(raw_objects, list):
        raise ValueError("LLM output field 'objects' is not a list.")

    parsed: List[Dict[str, Any]] = []
    seen_name = set()
    seen_text = set()

    for idx, obj in enumerate(raw_objects):
        if not isinstance(obj, dict):
            continue

        name = sanitize_object_name(
            obj.get("name", f"obj{idx}"),
            fallback=f"obj{idx}"
        )

        text = clean_text(obj.get("text", ""))
        if not text:
            continue

        # ---- 新增：解析 role / salience ----
        role = str(obj.get("role", "prop")).strip().lower()
        if role not in ROLE_ORDER:
            role = "prop"

        try:
            salience = int(obj.get("salience", 3))
        except Exception:
            salience = 3
        salience = max(1, min(5, salience))

        # ---- indices 校验（保持不变） ----
        active = obj.get("active_prompt_indices", [])
        if not isinstance(active, list):
            continue

        active_clean = []
        for x in active:
            try:
                k = int(x)
            except Exception:
                continue
            if 0 <= k < prompt_count:
                active_clean.append(k)

        active_clean = sorted(set(active_clean))
        if len(active_clean) == 0:
            continue

        if not allow_single_segment_object and len(active_clean) < min_active_segments:
            continue

        name_key = name.lower()
        text_key = text.lower()
        if name_key in seen_name or text_key in seen_text:
            continue
        seen_name.add(name_key)
        seen_text.add(text_key)

        parsed.append({
            "name": name,
            "text": text,
            "role": role,
            "salience": salience,
            "active_prompt_indices": active_clean,
        })

    if len(parsed) == 0:
        raise ValueError("No valid objects extracted.")

    # ---- 新增：按 role -> salience 降序 -> 覆盖段数降序 -> 首现段号升序 排序 ----
    parsed.sort(key=lambda o: (
        ROLE_ORDER[o["role"]],
        -o["salience"],
        -len(o["active_prompt_indices"]),
        o["active_prompt_indices"][0],
    ))

    # ---- 截断 + 剥离 role / salience，保证输出结构不变 ----
    objects = [
        {
            "name": o["name"],
            "text": o["text"],
            "active_prompt_indices": o["active_prompt_indices"],
        }
        for o in parsed[:max_objects]
    ]

    return {
        "prompt_duration": float(prompt_duration),
        "prompt_count": int(prompt_count),
        "objects": objects,
    }


# ============================================================
# 4. LLM 调用
# ============================================================

def build_client() -> OpenAI:
    if API_KEY is None or str(API_KEY).strip() == "":
        raise RuntimeError("API_KEY is empty. Please set OPENAI_API_KEY or write API_KEY directly.")

    if BASE_URL is not None and str(BASE_URL).strip() != "":
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)

    return OpenAI(api_key=API_KEY)


def parse_llm_content(content: str) -> Dict[str, Any]:
    content = content.strip()

    # 兼容某些模型偶尔输出 ```json ... ```
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    return json.loads(content)


def call_llm_for_object_config(
    client: OpenAI,
    prompts: List[str],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        prompts=prompts,
        prompt_duration=PROMPT_DURATION,
        max_objects=MAX_OBJECTS,
        min_active_segments=MIN_ACTIVE_SEGMENTS,
        allow_single_segment_object=ALLOW_SINGLE_SEGMENT_OBJECT,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    if RESPONSE_FORMAT_MODE == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": get_object_config_schema()
        }
    elif RESPONSE_FORMAT_MODE == "json_object":
        response_format = {
            "type": "json_object"
        }
    else:
        raise ValueError(f"Unsupported RESPONSE_FORMAT_MODE: {RESPONSE_FORMAT_MODE}")

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                response_format=response_format,
            )

            content = response.choices[0].message.content
            raw_cfg = parse_llm_content(content)
            return raw_cfg

        except Exception as e:
            last_error = e
            if attempt + 1 < MAX_RETRIES:
                sleep_time = RETRY_SLEEP * (attempt + 1)
                print(f"[Retry {attempt + 1}/{MAX_RETRIES}] {repr(e)}; sleep {sleep_time}s")
                time.sleep(sleep_time)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {repr(last_error)}")


# ============================================================
# 5. 主流程
# ============================================================

def process_one_record(
    client: OpenAI,
    line_idx: int,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    video_id = safe_video_id(record.get(VIDEO_ID_KEY), line_idx)
    video_path = record.get(VIDEO_PATH_KEY, None)

    prompts = extract_prompts(record, PROMPTS_KEY)
    prompt_count = len(prompts)

    raw_cfg = call_llm_for_object_config(
        client=client,
        prompts=prompts,
    )

    object_config = validate_and_normalize_config(
        raw_cfg=raw_cfg,
        prompt_count=prompt_count,
        prompt_duration=PROMPT_DURATION,
        max_objects=MAX_OBJECTS,
        min_active_segments=MIN_ACTIVE_SEGMENTS,
        allow_single_segment_object=ALLOW_SINGLE_SEGMENT_OBJECT,
    )

    object_config["video_id"] = video_id

    if video_path is not None:
        object_config["video_path"] = video_path

    object_config["source_line_idx"] = line_idx

    return object_config


def save_individual_object_config(video_id: str, object_config: Dict[str, Any]) -> str:
    out_path = os.path.join(OUTPUT_CONFIG_DIR, f"{video_id}_object_config.json")

    if os.path.exists(out_path) and not OVERWRITE_INDIVIDUAL_CONFIG:
        return out_path

    # ReferDINO 脚本实际只需要 prompt_duration / prompt_count / objects
    # 这里保留 video_id / video_path 也不影响读取
    write_json(out_path, object_config)

    return out_path

def _process_task(client, line_idx, record, video_id):
    """
    供线程池调用的包装函数。
    返回清洗后的 object_config 字典，或抛出异常。
    """
    try:
        object_config = process_one_record(client, line_idx, record)
        return object_config
    except Exception as e:
        # 异常会由 caller 捕获并记录
        raise

def main():
    import shutil

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_JSONL)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(FAILED_JSONL)), exist_ok=True)
    os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)

    done_ids = load_done_video_ids(OUTPUT_JSONL) if RESUME else set()

    client = build_client()

    total = 0
    success = 0
    skipped = 0
    failed = 0

    print("========== Batch LLM Object Config Extraction ==========")
    print(f"INPUT_JSONL       : {INPUT_JSONL}")
    print(f"OUTPUT_JSONL      : {OUTPUT_JSONL}")
    print(f"OUTPUT_CONFIG_DIR : {OUTPUT_CONFIG_DIR}")
    print(f"MODEL             : {MODEL}")
    print(f"PROMPT_DURATION   : {PROMPT_DURATION}")
    print(f"MAX_OBJECTS       : {MAX_OBJECTS}")
    print(f"RESUME            : {RESUME}")
    print(f"CONCURRENCY       : {CONCURRENCY}")
    print("========================================================")

    # 用于写失败日志的锁（append 是线程安全的，但为保险可加锁）
    fail_lock = threading.Lock()
    out_lock = threading.Lock()

    # 先收集所有需要处理的记录（支持断点续跑）
    tasks = []  # 每个元素 (line_idx, record, video_id)
    for line_idx, record, err in read_jsonl(INPUT_JSONL):
        total += 1

        if err is not None:
            failed += 1
            with fail_lock:
                append_jsonl(FAILED_JSONL, {
                    "line_idx": line_idx,
                    "error": err
                })
            print(f"[FAILED] line={line_idx}, error={err}")
            continue

        video_id = safe_video_id(record.get(VIDEO_ID_KEY), line_idx)

        if RESUME and video_id in done_ids:
            skipped += 1
            print(f"[SKIP] {video_id}")
            continue

        tasks.append((line_idx, record, video_id))

    # 使用线程池并发调用 LLM
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # 提交所有任务，每个任务返回 (line_idx, object_config 或 None, 异常信息)
        future_to_task = {}
        for line_idx, record, video_id in tasks:
            future = executor.submit(
                _process_task,
                client, line_idx, record, video_id
            )
            future_to_task[future] = (line_idx, record, video_id)

        # 按完成顺序处理结果（顺序不重要，因为每条记录独立）
        for future in as_completed(future_to_task):
            line_idx, record, video_id = future_to_task[future]
            try:
                result = future.result()
                if result is None:
                    # 任务返回 None 表示内部已记录失败（由 _process_task 处理）
                    continue

                object_config = result  # 已清洗并验证的配置

                # 保存单个配置 JSON
                individual_path = save_individual_object_config(
                    video_id=video_id,
                    object_config=object_config,
                )

                output_record = {
                    "video_id": video_id,
                    "video_path": record.get(VIDEO_PATH_KEY, None),
                    "object_config_path": individual_path,
                    "object_config": object_config
                }

                with out_lock:
                    append_jsonl(OUTPUT_JSONL, output_record)

                success += 1
                obj_summary = [
                    f"{obj['name']}:{obj['active_prompt_indices']}"
                    for obj in object_config["objects"]
                ]
                print(f"[OK] {video_id} -> {individual_path} | objects={obj_summary}")

            except Exception as e:
                failed += 1
                fail_record = {
                    "line_idx": line_idx,
                    "video_id": video_id,
                    "video_path": record.get(VIDEO_PATH_KEY, None),
                    "error": repr(e)
                }
                with fail_lock:
                    append_jsonl(FAILED_JSONL, fail_record)
                print(f"[FAILED] {video_id}, error={repr(e)}")

    print("========== Done ==========")
    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Output  : {OUTPUT_JSONL}")
    print(f"Failed  : {FAILED_JSONL}")


if __name__ == "__main__":
    main()