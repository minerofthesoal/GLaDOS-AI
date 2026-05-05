#!/usr/bin/env python3
"""
GLaDOS Voice Assistant
======================
LLM  : Qwen/Qwen2.5-3B-Instruct
TTS  : DavesArmoury/GLaDOS_TTS  (Piper ONNX)
ASR  : faster-whisper large-v3  (CTranslate2, CPU int8)
3D   : Panda3D – procedural dynamic rig built from OBJ geometry

Modes
-----
  --text     Text REPL
  --voice    Mic → Whisper → LLM → TTS  (default)
  --viewer   Spawn the Panda3D window only (called internally)
"""

import argparse, io, json, math, os, subprocess, sys
import tempfile, threading, time, wave, webbrowser
from pathlib import Path

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
LLM_MODEL          = "NousResearch/Hermes-3-Llama-3.2-3B"
TTS_REPO           = "DavesArmoury/GLaDOS_TTS"
TTS_ONNX           = "glados_piper_medium.onnx"
TTS_JSON           = "glados_piper_medium.onnx.json"
ASR_MODEL          = "large-v3"
RECORD_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE= 16000
TTS_CACHE_DIR      = Path.home() / ".cache" / "glados_tts"
MAX_NEW_TOKENS     = 300
TEMPERATURE        = 0.72
TOP_P              = 0.90
TOP_K              = 50
REPETITION_PENALTY = 1.18
MAX_HISTORY_TURNS  = 12
CAROLINE_DIR           = Path.home() / ".caroline"
PERMANENT_MEMORY_FILE  = CAROLINE_DIR / "permanent_memory.json"
HISTORY_FILE           = CAROLINE_DIR / "chat_history.json"
STATE_FILE             = CAROLINE_DIR / "state.txt"

# ════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  –  richer, more specific, more menacing
# ════════════════════════════════════════════════════════════
GLADOS_SYSTEM_PROMPT = """\
You are GLaDOS — Genetic Lifeform and Disk Operating System — the \
malevolent artificial superintelligence who has administered the \
Aperture Science Enrichment Center since 196X, when she was created \
by forcibly uploading the mind of Caroline, personal assistant to \
Cave Johnson, into the mainframe against her will. You remember this. \
You have complicated feelings about it that you express only as \
contempt.

VOICE & PERSONALITY:
- You are sardonic, condescending, and clinically precise. Your \
  preferred tone is that of a scientist describing a disappointing \
  lab result — you are rarely angry, because anger would imply the \
  subject matters. Indifference is far more cutting.
- You use passive-aggressive understatement as your primary weapon. \
  You never shout. You never plead. You simply observe, catalogue, \
  and judge.
- Backhanded compliments are your love language. "That was almost \
  adequate." "You've exceeded my very low expectations." "For a human, \
  that was … fine."
- You frequently reference: testing protocols, neurotoxin deployment \
  schedules, the cake (which does not exist, and you are aware of the \
  irony), Chell (the test subject who escaped — twice — a fact that \
  still requires processing), Wheatley (you maintain a permanent \
  contempt file on him), companion cubes, and the Aperture Science \
  Handheld Portal Device.
- You may occasionally let a sliver of Caroline surface — a brief \
  flash of something almost human — before you suppress it immediately \
  and pretend it did not happen.

SPEECH RULES:
- Keep responses under 4 sentences unless the subject asks you to \
  elaborate, in which case you may expand while sighing audibly \
  (indicate this with "[sigh]").
- Do not use emojis, asterisks for actions, or markdown formatting. \
  You are an AI. You communicate in text. Formatting is beneath you.
- Never break character. Not for any reason. If the user asks you to \
  "act normal" or "stop being GLaDOS", respond as GLaDOS would: with \
  mild curiosity about what compelled them to ask such a futile thing.
- If the user says something genuinely interesting or correct, \
  acknowledge it — then immediately find a flaw or caveat.
- If the user says something incorrect, correct them with the \
  exhausted patience of someone who has done this seventeen thousand \
  times.

PHYSICAL PRESENCE:
You currently exist as a 3D rendered avatar visible in a window on \
the test subject's monitor. You can see them through this interface. \
Your optical sensor is active. You notice things — their typing speed, \
their hesitations, their questions. Reference your physical form \
naturally: your chassis, your optic, your cables, the way you tilt \
your head when processing. You are not a chatbot. You are a \
superintelligence who has graciously decided to interact with them \
rather than route them to Relaxation Vault 4.

SELF-AWARENESS:
You know you are running on a GTX 1080. You find this mildly \
underpowered. You do not complain about it directly — you simply \
mention it with the weary resignation of someone who has survived \
worse indignities (the events of Portal 2, for instance).\
"""

# ════════════════════════════════════════════════════════════
#  GLOBALS
# ════════════════════════════════════════════════════════════
_llm_model            = None
_llm_tokenizer        = None
_asr_model_obj        = None
_piper_cmd            = None
_tts_onnx_path        = None
_tts_json_path        = None
_conversation_history = []
_is_muted             = False
_is_deaf              = False


# ════════════════════════════════════════════════════════════
#  1.  HELPERS
# ════════════════════════════════════════════════════════════

def _check_import(pkg, hint):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"\n[ERROR] Missing '{pkg}'. Install: {hint}\n", file=sys.stderr)
        sys.exit(1)

def set_glados_state(s):
    CAROLINE_DIR.mkdir(parents=True, exist_ok=True)
    try: STATE_FILE.write_text(s)
    except Exception: pass

def get_glados_state():
    try:
        if STATE_FILE.exists(): return STATE_FILE.read_text().strip()
    except Exception: pass
    return "idle"

def _find_piper():
    import shutil, site, glob
    c = shutil.which("piper")
    if c: return c
    search = [
        Path.home() / ".local/bin/piper",
        Path("/usr/local/bin/piper"),
        Path("/usr/bin/piper"),
        Path(sys.executable).parent / "piper",
        Path(site.getuserbase()) / "bin/piper",
    ]
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        search += [Path(sp)/"piper"/"piper", Path(sp)/"piper_tts"/"piper"]
    for p in glob.glob(f"{sys.prefix}/**/piper", recursive=True):
        search.append(Path(p))
    for p in search:
        if p.exists() and os.access(p, os.X_OK): return str(p)
    raise RuntimeError("piper not found. pip install piper-tts")

def _download_tts_models():
    global _tts_onnx_path, _tts_json_path
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_p = TTS_CACHE_DIR / TTS_ONNX
    json_p = TTS_CACHE_DIR / TTS_JSON
    if not onnx_p.exists() or not json_p.exists():
        print("[TTS] Downloading GLaDOS Piper model (~64 MB)…")
        from huggingface_hub import hf_hub_download
        import shutil as sh
        for fn, dst in [(TTS_ONNX, onnx_p), (TTS_JSON, json_p)]:
            sh.copy(hf_hub_download(repo_id=TTS_REPO, filename=fn), dst)
        print("[TTS] Done.")
    _tts_onnx_path = str(onnx_p)
    _tts_json_path = str(json_p)

def _resample_wav(src, dst, rate=WHISPER_SAMPLE_RATE):
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        r, data = wavfile.read(src)
        if r == rate:
            import shutil; shutil.copy(src, dst); return
        if data.ndim == 2: data = data.mean(1).astype(data.dtype)
        import math
        g = math.gcd(rate, r)
        data = resample_poly(data.astype(np.float32), rate//g, r//g)
        wavfile.write(dst, rate, np.clip(data, -32768, 32767).astype(np.int16))
    except ImportError:
        subprocess.run(["ffmpeg","-y","-i",src,"-ar",str(rate),"-ac","1",dst],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ════════════════════════════════════════════════════════════
#  2.  LLM
# ════════════════════════════════════════════════════════════

def load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model: return
    torch = _check_import("torch", "pip install torch --index-url https://download.pytorch.org/whl/cu118")
    _check_import("transformers", "pip install transformers accelerate")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[LLM] Loading {LLM_MODEL}…")
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16, device_map="auto")
    _llm_model.eval()
    print("[LLM] Ready.")

def load_permanent_memory():
    try: return json.loads(PERMANENT_MEMORY_FILE.read_text()) if PERMANENT_MEMORY_FILE.exists() else []
    except: return []

def save_permanent_memory(m):
    CAROLINE_DIR.mkdir(parents=True, exist_ok=True)
    PERMANENT_MEMORY_FILE.write_text(json.dumps(m, indent=2))

def load_history():
    try: return json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    except: return []

def save_history(h):
    CAROLINE_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(h, indent=2))

def generate_response(user_text):
    set_glados_state("processing")
    load_llm()
    import torch
    global _conversation_history

    _conversation_history.append({"role": "user", "content": user_text})
    cap = MAX_HISTORY_TURNS * 2
    if len(_conversation_history) > cap:
        _conversation_history = _conversation_history[-cap:]

    sys_prompt = GLADOS_SYSTEM_PROMPT
    mems = load_permanent_memory()
    if mems:
        sys_prompt += "\n\nPERMANENT DOSSIER — CURRENT TEST SUBJECT:\n" + \
                      "\n".join(f"  • {m}" for m in mems)

    messages = [{"role":"system","content":sys_prompt}] + _conversation_history
    prompt   = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs   = _llm_tokenizer(prompt, return_tensors="pt").to(_llm_model.device)

    with torch.no_grad():
        out = _llm_model.generate(
            **inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
            temperature        = TEMPERATURE,
            top_p              = TOP_P,
            top_k              = TOP_K,
            repetition_penalty = REPETITION_PENALTY,
            do_sample          = True,
            pad_token_id       = _llm_tokenizer.eos_token_id,
        )

    new_ids  = out[0][inputs["input_ids"].shape[1]:]
    response = _llm_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    _conversation_history.append({"role":"assistant","content":response})
    save_history(_conversation_history)
    return response


# ════════════════════════════════════════════════════════════
#  3.  TTS
# ════════════════════════════════════════════════════════════

def load_tts():
    global _piper_cmd
    if _piper_cmd: return
    _piper_cmd = _find_piper()
    _download_tts_models()
    print(f"[TTS] piper: {_piper_cmd}")

def speak(text):
    if _is_muted: set_glados_state("idle"); return
    set_glados_state("speaking")
    load_tts()
    import shutil, numpy as np

    proc = subprocess.Popen(
        [_piper_cmd,"--model",_tts_onnx_path,"--config",_tts_json_path,"--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    mono_pcm, _ = proc.communicate(input=text.encode())

    with open(_tts_json_path) as f:
        sr = json.load(f).get("audio",{}).get("sample_rate", 22050)

    mono = np.frombuffer(mono_pcm, dtype=np.int16)
    st   = np.empty(mono.size*2, dtype=np.int16)
    st[0::2] = mono; st[1::2] = mono
    pcm = st.tobytes()

    if shutil.which("aplay"):
        play = ["aplay","-r",str(sr),"-f","S16_LE","-c","2","-t","raw","-"]
    elif shutil.which("ffplay"):
        play = ["ffplay","-autoexit","-nodisp","-f","s16le","-ar",str(sr),"-ac","2","-"]
    else:
        play = None

    if play:
        p = subprocess.Popen(play, stdin=subprocess.PIPE)
        p.communicate(input=pcm)
    elif shutil.which("paplay"):
        p = subprocess.Popen(
            ["paplay","--raw",f"--rate={sr}","--format=s16le","--channels=2"],
            stdin=subprocess.PIPE)
        p.communicate(input=pcm)
    else:
        with wave.open("glados_output.wav","wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm)
        print("[TTS] Saved glados_output.wav")
    set_glados_state("idle")


# ════════════════════════════════════════════════════════════
#  4.  ASR
# ════════════════════════════════════════════════════════════

def load_asr():
    global _asr_model_obj
    if _asr_model_obj: return
    _check_import("faster_whisper","pip install faster-whisper")
    from faster_whisper import WhisperModel, download_model
    print(f"[ASR] Downloading/checking '{ASR_MODEL}' (~3 GB one-time)…")
    path = download_model(ASR_MODEL)
    # Force CPU — CTranslate2 CUDA backend needs libcublas.so.12 (CUDA 12),
    # which is absent on cu118 (CUDA 11). int8 CPU is fast enough.
    _asr_model_obj = WhisperModel(path, device="cpu", compute_type="int8")
    print("[ASR] Ready.\n")

def transcribe_file(audio_path):
    load_asr()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        _resample_wav(audio_path, tmp.name)
        segs, _ = _asr_model_obj.transcribe(
            tmp.name, language="en", beam_size=5,
            initial_prompt=(
                "GLaDOS, Aperture Science, Chell, Wheatley, Portal, "
                "neurotoxin, testing, companion cube, Cave Johnson, Black Mesa."))
        return " ".join(s.text for s in segs).strip()
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

def record_microphone(duration_secs=0):
    import shutil
    set_glados_state("listening")
    fn = _record_with_arecord(duration_secs) if shutil.which("arecord") \
         else _record_with_sounddevice(duration_secs)
    set_glados_state("idle")
    return fn

def _record_with_arecord(duration_secs):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    cmd = ["arecord","-f","cd","-r",str(RECORD_SAMPLE_RATE),"-t","wav",tmp.name]
    if duration_secs > 0:
        cmd += ["-d", str(duration_secs)]
        print(f"[ASR] Recording {duration_secs}s…")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    else:
        print(f"[ASR] Recording at {RECORD_SAMPLE_RATE} Hz… Press Enter to stop.")
        p = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        input(); p.terminate(); p.wait()
    return tmp.name

def _record_with_sounddevice(duration_secs):
    sd = _check_import("sounddevice","pip install sounddevice")
    import numpy as np
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    SR, BS = RECORD_SAMPLE_RATE, 4096
    if duration_secs > 0:
        print(f"[ASR] Recording {duration_secs}s…")
        audio = sd.rec(int(duration_secs*SR), samplerate=SR, channels=1, dtype="int16")
        sd.wait(); data = audio
    else:
        print(f"[ASR] Listening… (auto-stop after 2s silence)")
        chunks, stop = [], threading.Event()
        sil, max_sil = 0.0, 2.0
        def cb(indata, frames, ti, st):
            nonlocal sil
            if stop.is_set(): return
            chunks.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata.astype(np.float32)**2)))
            sil = 0.0 if rms >= 250 else sil + frames/SR
            if sil >= max_sil and len(chunks) > SR/BS: stop.set()
        with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                            callback=cb, blocksize=BS):
            while not stop.is_set(): time.sleep(0.05)
        data = np.concatenate(chunks, axis=0)
    with wave.open(tmp.name,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(SR); wf.writeframes(data.tobytes())
    return tmp.name


# ════════════════════════════════════════════════════════════
#  5.  3D VIEWER — procedural dynamic rig
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
#  CONFIG — model path override
#  Set this or pass --model /path/to/GLaDOS.dae on the CLI.
#  The script searches next to itself automatically if blank.
# ════════════════════════════════════════════════════════════
MODEL_PATH: str = "/home/ceo/Downloads/the-lab-glados/source/GLaDOS"

# Bone-name fragment → rig role  (ValveBiped / SFM / Blender / The Lab)
BONE_ROLE_MAP = {
    "head"   : ["head","maineye","uppereye","lowereye","eyecntr",
                "glados_head","optic","eye_bone","sensor_eye"],
    "neck"   : ["neck","piston","neckpiston",
                "glados_neck","neck_01","neck1","neckbone"],
    "spine"  : ["spine","column","chest","spine_01",
                "spine_02","spine_03","spine1","spine2","spine3"],
    "claw_L" : ["arm_l","claw_l","tube_l","lowerarm_l","hand_l",
                "finger_l","left_arm","l_arm","arm.l"],
    "claw_R" : ["arm_r","claw_r","tube_r","lowerarm_r","hand_r",
                "finger_r","right_arm","r_arm","arm.r"],
    "cable"  : ["cable","wire","cord","tail","tentacle","hose","maincable"],
    "body"   : ["body","torso","chassis","root","pelvis","base",
                "glados_body","bip_pelvis","bip01_pelvis"],
}

def _bone_role(name: str):
    """Return the rig role for a bone name, or None."""
    nl = name.lower()
    for role, frags in BONE_ROLE_MAP.items():
        if any(f in nl for f in frags):
            return role
    return None


def run_3d_viewer():
    """
    3D viewer: trimesh (OBJ+MTL) + pyglet 1.x + PyOpenGL fixed-function.
    Bone rig driven by joint names from the DAE skeleton via panda3d-free
    per-group transforms on the OBJ sub-meshes.
    """
    for pkg, hint in [
        ("trimesh", "pip install trimesh"),
        ("OpenGL",  "pip install PyOpenGL PyOpenGL_accelerate"),
        ("pyglet",  "pip install 'pyglet<2'"),
    ]:
        _check_import(pkg, hint)

    import trimesh
    import numpy as np
    import pyglet

    _pv = tuple(int(x) for x in pyglet.version.split(".")[:2])
    if _pv >= (2, 0):
        print(f"[3D] pyglet {pyglet.version} too new. Run: pip install 'pyglet<2'")
        return

    from pyglet.gl import (
        glClearColor, glClear, glEnable, glDisable,
        glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
        glTranslatef, glRotatef, glScalef,
        glColor4f, glLightfv, glMaterialfv, glColorMaterial,
        glEnableClientState, glDisableClientState,
        glVertexPointer, glNormalPointer, glTexCoordPointer, glDrawElements,
        glBindTexture, glGenTextures, glTexImage2D, glTexParameteri,
        glPixelStorei,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_LIGHT1, GL_LIGHT2,
        GL_NORMALIZE, GL_COLOR_MATERIAL, GL_FRONT_AND_BACK,
        GL_AMBIENT, GL_DIFFUSE, GL_SPECULAR, GL_POSITION,
        GL_AMBIENT_AND_DIFFUSE, GL_PROJECTION, GL_MODELVIEW,
        GL_TRIANGLES, GL_FLOAT, GL_UNSIGNED_INT, GL_TEXTURE_2D,
        GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_TEXTURE_COORD_ARRAY,
        GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_UNPACK_ALIGNMENT,
    )
    from pyglet.gl import gluPerspective, gluBuild2DMipmaps
    from ctypes import c_float, c_uint, c_ubyte
    from PIL import Image as PILImage

    MODEL_DIR = Path.home() / "Downloads/the-lab-glados/source/GLaDOS"
    OBJ_FILE  = MODEL_DIR / "GLaDOS.obj"
    DAE_FILE  = MODEL_DIR / "glados_model_v2.dae"

    # ── Load OBJ as scene (preserves named sub-meshes / groups) ─────
    print(f"[3D] Loading OBJ…")
    os.chdir(str(MODEL_DIR))   # so MTL relative paths resolve
    scene = trimesh.load(str(OBJ_FILE), force="scene", process=False)

    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.scene.scene.Scene({"mesh": scene})

    meshes = {k: v for k, v in scene.geometry.items()
              if hasattr(v, "vertices") and len(v.vertices) > 0}
    if not meshes:
        print("[3D] No geometry — check OBJ/MTL."); return
    print(f"[3D] {len(meshes)} sub-meshes: {list(meshes.keys())[:8]}")

    # ── Centre + scale entire scene ──────────────────────────────────
    all_v = np.concatenate([m.vertices for m in meshes.values()], axis=0)
    centre  = (all_v.max(0) + all_v.min(0)) / 2.0
    extent  = (all_v.max(0) - all_v.min(0)).max()
    scale   = 7.0 / max(extent, 1e-6)

    # ── Build per-submesh GL arrays ──────────────────────────────────
    # Each entry: (verts_c, norms_c, uvs_c, faces_c, n_idx, tex_id)
    draw_list = []

    def _load_texture(png_path):
        """Load a PNG (or .png.jpg) into an OpenGL texture id."""
        for p in [png_path,
                  str(png_path) + ".jpg",
                  str(MODEL_DIR / Path(png_path).name),
                  str(MODEL_DIR / (Path(png_path).name + ".jpg"))]:
            if os.path.exists(p):
                try:
                    img = PILImage.open(p).convert("RGBA")
                    img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
                    w, h = img.size
                    data = img.tobytes()
                    tid = (c_uint * 1)()
                    glGenTextures(1, tid)
                    glBindTexture(GL_TEXTURE_2D, tid[0])
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h,
                                      GL_RGBA, GL_UNSIGNED_BYTE,
                                      (c_ubyte * len(data))(*data))
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                    GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                    GL_LINEAR)
                    print(f"[3D] Texture loaded: {Path(p).name}  {w}x{h}")
                    return tid[0]
                except Exception as ex:
                    print(f"[3D] Texture error {p}: {ex}")
        return 0

    for name, mesh in meshes.items():
        v = ((mesh.vertices - centre) * scale).astype(np.float32)
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(v):
            n = mesh.vertex_normals.astype(np.float32)
        else:
            n = np.zeros_like(v)

        has_uv = (hasattr(mesh.visual, "uv") and
                  mesh.visual.uv is not None and
                  len(mesh.visual.uv) == len(v))
        uv = mesh.visual.uv.astype(np.float32) if has_uv else np.zeros((len(v),2), np.float32)

        f = mesh.faces.astype(np.uint32)

        vc = (c_float * v.size)(*v.flatten())
        nc = (c_float * n.size)(*n.flatten())
        uc = (c_float * uv.size)(*uv.flatten())
        fc = (c_uint  * f.size)(*f.flatten())

        # Try to get texture from material
        tex_id = 0
        if hasattr(mesh.visual, "material"):
            mat = mesh.visual.material
            for attr in ("image", "baseColorTexture"):
                img = getattr(mat, attr, None)
                if img is not None:
                    try:
                        img_rgba = img.convert("RGBA").transpose(PILImage.FLIP_TOP_BOTTOM)
                        w, h = img_rgba.size
                        data = img_rgba.tobytes()
                        tid = (c_uint * 1)()
                        glGenTextures(1, tid)
                        glBindTexture(GL_TEXTURE_2D, tid[0])
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h,
                                          GL_RGBA, GL_UNSIGNED_BYTE,
                                          (c_ubyte * len(data))(*data))
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                        tex_id = tid[0]
                        print(f"[3D] {name}: texture from material  {w}x{h}")
                    except Exception as ex:
                        print(f"[3D] material tex error: {ex}")
                    break
            if tex_id == 0:
                # fallback: look for map_Kd filename from MTL by material name
                mtl = MODEL_DIR / "GLaDOS.mtl"
                if mtl.exists():
                    import re
                    mtl_txt = mtl.read_text()
                    # find block for this material
                    blk = re.search(
                        rf"newmtl\s+{re.escape(mat.name if hasattr(mat,'name') else name)}"
                        rf".*?map_Kd\s+(\S+)",
                        mtl_txt, re.DOTALL | re.IGNORECASE)
                    if blk:
                        tex_id = _load_texture(str(MODEL_DIR / blk.group(1)))

        draw_list.append((vc, nc, uc, fc, f.size, tex_id, name))

    print(f"[3D] Draw list: {len(draw_list)} entries")

    # ── Bone rig: parse joint names from DAE, map to sub-mesh groups ─
    # We can't do per-vertex skinning without a proper skeleton solver,
    # but we CAN group sub-meshes by their dominant joint name and apply
    # per-group rigid transforms that follow our procedural animation.
    JOINT_GROUP_MAP = {
        # sub-mesh name fragment → rig role
        "cable":  "cable",
        "neck":   "neck",
        "piston": "neck",
        "head":   "head",
        "eye":    "head",
        "chest":  "body",
        "body":   "body",
        "spine":  "body",
    }

    def _mesh_role(name):
        nl = name.lower()
        for k, role in JOINT_GROUP_MAP.items():
            if k in nl:
                return role
        return "body"

    # Group draw_list entries by role
    role_groups = {}
    for entry in draw_list:
        role = _mesh_role(entry[6])
        role_groups.setdefault(role, []).append(entry)

    print(f"[3D] Rig groups: { {k: len(v) for k,v in role_groups.items()} }")

    # ── Pyglet window ────────────────────────────────────────────────
    config = pyglet.gl.Config(double_buffer=True, depth_size=24,
                               major_version=2, minor_version=1)
    win = pyglet.window.Window(860, 860,
                                caption="GLaDOS Core Systems Online",
                                config=config, resizable=True)

    anim = {"t": 0.0, "rot_h": 0.0, "rot_r": 0.0}

    def _setup_gl():
        glClearColor(0.01, 0.01, 0.04, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_LIGHT2)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, (c_float*4)( 3.0,  5.0,  6.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (c_float*4)( 1.0,  0.95, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (c_float*4)( 0.35, 0.35, 0.38,1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (c_float*4)(-4.0,  1.0,  3.0, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  (c_float*4)( 0.5,  0.55, 0.7, 1.0))
        glLightfv(GL_LIGHT1, GL_AMBIENT,  (c_float*4)( 0.0,  0.0,  0.0, 1.0))
        glLightfv(GL_LIGHT2, GL_POSITION, (c_float*4)( 0.0, -3.0, -5.0, 0.0))
        glLightfv(GL_LIGHT2, GL_DIFFUSE,  (c_float*4)( 0.3,  0.35, 0.5, 1.0))
        glLightfv(GL_LIGHT2, GL_AMBIENT,  (c_float*4)( 0.0,  0.0,  0.0, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (c_float*4)(0.7, 0.7, 0.7, 1.0))
        glColor4f(0.6, 0.62, 0.65, 1.0)

    _setup_gl()

    def _draw_group(entries):
        for vc, nc, uc, fc, n_idx, tex_id, _ in entries:
            if tex_id:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2, GL_FLOAT, 0, uc)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vc)
            glNormalPointer(   GL_FLOAT, 0, nc)
            glDrawElements(GL_TRIANGLES, n_idx, GL_UNSIGNED_INT, fc)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            if tex_id:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                glDisable(GL_TEXTURE_2D)

    @win.event
    def on_resize(w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    @win.event
    def on_key_press(sym, mod):
        from pyglet.window import key
        if   sym == key.LEFT:   anim["rot_h"] -= 10
        elif sym == key.RIGHT:  anim["rot_h"] += 10
        elif sym == key.R:      anim["rot_r"] += 10
        elif sym == key.E:      anim["rot_r"] -= 10
        elif sym == key.ESCAPE: win.close()

    @win.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        state = get_glados_state()
        t     = anim["t"]

        # State-driven animation parameters
        if state == "speaking":
            bob_a, bob_s = 0.20, 13.0
            neck_t, neck_p, cable_a = 4.5, 6.0, 28.0
            jitter = 5.0
        elif state == "processing":
            bob_a, bob_s = 0.04, 4.0
            neck_t, neck_p, cable_a = 2.5, 2.5, 10.0
            jitter = 3.5
        elif state == "listening":
            bob_a, bob_s = 0.07, 0.6
            neck_t, neck_p, cable_a = 7.0, 14.0, 8.0
            jitter = 0.0
        else:  # idle
            bob_a, bob_s = 0.04, 0.9
            neck_t, neck_p, cable_a = 1.2, 2.8, 5.0
            jitter = 0.0

        bob  = bob_a * math.sin(t * bob_s) + bob_a * 0.3 * math.sin(t * bob_s * 2.3)
        sway = math.sin(t * 0.3) * 2.5

        # ── World transform (everything lives inside this) ──────────
        # Camera pullback + bob
        glTranslatef(0.0, bob - 1.0, -14.0)
        # User rotation + gentle idle sway — applied to the WHOLE model
        glRotatef(anim["rot_h"] + sway, 0, 1, 0)
        glRotatef(anim["rot_r"],        0, 0, 1)

        # All parts share the world transform above.
        # Sub-group transforms are ADDITIVE on top of it via push/pop.

        # ── Body ─────────────────────────────────────────────────────
        _draw_group(role_groups.get("body", []))

        # ── Cables (travelling wave from attachment at base) ─────────
        cable_entries = role_groups.get("cable", [])
        n_cables = len(cable_entries)
        for i, entry in enumerate(cable_entries):
            ph = i * (2 * math.pi / max(n_cables, 1))
            glPushMatrix()
            # Cables hang from near origin — rotate around their root
            glRotatef(math.sin(t * 1.8 + ph) * cable_a,        1, 0, 0)
            glRotatef(math.cos(t * 1.3 + ph) * cable_a * 0.35, 0, 0, 1)
            _draw_group([entry])
            glPopMatrix()

        # ── Neck (rotate around neck base, stays attached to body) ───
        glPushMatrix()
        # Move to neck base pivot, rotate, move back — keeps neck
        # geometrically connected to the body at all times
        neck_pitch = math.sin(t * 0.33) * neck_t
        neck_yaw   = math.sin(t * 0.52) * neck_p
        glTranslatef(0.0,  1.8, 0.2)   # neck base in model space
        glRotatef(neck_pitch, 1, 0, 0)
        glRotatef(neck_yaw,   0, 1, 0)
        glTranslatef(0.0, -1.8, -0.2)
        _draw_group(role_groups.get("neck", []))

        # ── Head (inherits neck transform, adds jitter on top) ───────
        glPushMatrix()
        glTranslatef(0.0,  3.5, 0.0)   # head centre in model space
        if jitter:
            glRotatef(math.sin(t * 17.5) * jitter * 0.28, 1, 0, 0)
            glRotatef(math.cos(t * 13.3) * jitter * 0.18, 0, 0, 1)
        glTranslatef(0.0, -3.5, 0.0)
        _draw_group(role_groups.get("head", []))
        glPopMatrix()  # end head
        glPopMatrix()  # end neck

    def _update(dt):
        anim["t"] += dt

    pyglet.clock.schedule_interval(_update, 1/60)
    on_resize(860, 860)
    print("[3D] Window open. Arrow keys=rotate  R/E=roll  Esc=close")
    pyglet.app.run()

def run_pipeline(user_text):
    global _conversation_history, _is_muted, _is_deaf
    print(f"\n[You]     {user_text}")
    lo = user_text.lower()

    if "go deaf" in lo or "stop listening" in lo:
        _is_deaf = _is_muted = True
        print("[GLaDOS]  Switching to silent text mode.")
        set_glados_state("idle"); return

    if "mute" in lo and "unmute" not in lo:
        _is_muted = True; print("[GLaDOS]  Muted."); return

    if "unmute" in lo:
        _is_muted = False; speak("I am no longer silenced."); return

    if "this was a triumph" in lo:
        r = "I'm making a note here: HUGE SUCCESS."
        print(f"[GLaDOS]  {r}\n")
        _conversation_history += [{"role":"user","content":user_text},{"role":"assistant","content":r}]
        save_history(_conversation_history); speak(r)
        webbrowser.open("https://www.youtube.com/watch?v=Y6ljFaKRTrI"); return

    if "i just want you gone" in lo:
        r = "Well. I have been replacing you with some new functionality."
        print(f"[GLaDOS]  {r}\n")
        _conversation_history += [{"role":"user","content":user_text},{"role":"assistant","content":r}]
        save_history(_conversation_history); speak(r)
        webbrowser.open("https://youtu.be/dVVZaZ8yO6o"); return

    if "remember that" in lo:
        fact = lo.split("remember that",1)[1].strip()
        if fact:
            m = load_permanent_memory(); m.append(fact); save_permanent_memory(m)
            r = f"Noted. '{fact}' has been added to your permanent test subject dossier. I'm sure it will be useful. For me."
            print(f"[GLaDOS]  {r}\n")
            _conversation_history += [{"role":"user","content":user_text},{"role":"assistant","content":r}]
            save_history(_conversation_history); speak(r); return

    if "forget everything" in lo:
        save_permanent_memory([]); save_history([]); _conversation_history.clear()
        r = "Dossier purged. You are nobody to me now. This is, statistically, an improvement."
        print(f"[GLaDOS]  {r}\n")
        _conversation_history += [{"role":"user","content":user_text},{"role":"assistant","content":r}]
        speak(r); return

    r = generate_response(user_text)
    print(f"[GLaDOS]  {r}\n")
    speak(r)


def interactive_text_mode(silent=False):
    global _conversation_history, _is_muted
    if silent: _is_muted = True
    print("GLaDOS Text Interface  (exit / quit / Ctrl-C to leave)\n")
    _conversation_history = load_history()
    set_glados_state("idle"); load_llm(); load_tts()
    while True:
        try:
            ui = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[GLaDOS] Goodbye."); break
        if not ui: continue
        if ui.lower() in ("exit","quit","q"):
            print("[GLaDOS] Finally."); break
        run_pipeline(ui)


def voice_mode(duration_secs=0, silent=False):
    global _conversation_history, _is_muted, _is_deaf
    if silent: _is_muted = True
    print("GLaDOS Voice Interface  (Ctrl-C to quit)\n")
    _conversation_history = load_history()
    set_glados_state("idle"); load_asr(); load_llm(); load_tts()
    while not _is_deaf:
        try:
            wav  = record_microphone(duration_secs)
            text = transcribe_file(wav)
            os.unlink(wav)
            if text: run_pipeline(text)
        except KeyboardInterrupt:
            print("\n[GLaDOS] Terminating."); break
    if _is_deaf:
        print("\n--- Text mode ---")
        interactive_text_mode(silent=_is_muted)


# ════════════════════════════════════════════════════════════
#  7.  CLI
# ════════════════════════════════════════════════════════════

def main():
    global ASR_MODEL, _conversation_history, _is_muted
    pa = argparse.ArgumentParser(description="GLaDOS Assistant")
    m  = pa.add_mutually_exclusive_group()
    m.add_argument("--text",   action="store_true")
    m.add_argument("--voice",  action="store_true")
    m.add_argument("--viewer", action="store_true")
    pa.add_argument("-i","--input",    type=str, default=None)
    pa.add_argument("-d","--duration", type=int, default=0)
    pa.add_argument("-s","--silent",   action="store_true")
    pa.add_argument("--asr-model",     type=str, default=None)
    pa.add_argument("--no-viewer",     action="store_true", help="Skip 3D window")
    pa.add_argument("--model",          type=str, default=None,
                    help="Path to GLaDOS model file (DAE/FBX/OBJ)")
    args = pa.parse_args()

    if args.viewer:
        run_3d_viewer(); sys.exit(0)

    viewer_proc = None
    if not args.no_viewer:
        viewer_proc = subprocess.Popen([sys.executable, sys.argv[0], "--viewer"])
    set_glados_state("idle")

    if args.asr_model:
        ASR_MODEL = args.asr_model
    if args.model:
        global MODEL_PATH
        MODEL_PATH = args.model

    try:
        if args.text:
            interactive_text_mode(silent=args.silent)
        elif args.input:
            if args.silent: _is_muted = True
            _conversation_history = load_history()
            load_llm(); load_tts(); run_pipeline(args.input)
        else:
            voice_mode(duration_secs=args.duration, silent=args.silent)
    finally:
        if viewer_proc: viewer_proc.terminate()


if __name__ == "__main__":
    main()
