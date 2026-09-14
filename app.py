"""Startup Crew — local control panel.

Run with:
    .venv/Scripts/streamlit run app.py
"""

import io
import json
import os
import sys
import uuid
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml
from dotenv import load_dotenv, set_key

# --------------------------------------------------------------------------
# Python version guard.
#
# CrewAI declares requires-python <3.14, and chromadb (pulled in by CrewAI)
# still builds its Settings through pydantic's v1 compatibility shim, which
# cannot infer field types under Python 3.14. The result is a cryptic
#   pydantic.v1.errors.ConfigError: unable to infer type for attribute
#   "chroma_server_nofile"
# raised deep inside an import. Downgrading pydantic does NOT help - the
# break is in 3.14 itself. Fail loudly and early instead.
# --------------------------------------------------------------------------
if sys.version_info >= (3, 14):
    st.error(
        f"This app needs Python 3.13 or older - it is running on "
        f"{sys.version_info.major}.{sys.version_info.minor}.\n\n"
        "CrewAI does not support Python 3.14 (it declares requires-python "
        "<3.14), and its chromadb dependency crashes on import with "
        '`unable to infer type for attribute \"chroma_server_nofile\"`.'
        "\n\nOn Streamlit Community Cloud the Python version cannot be "
        "changed after deployment: delete the app and redeploy it, choosing "
        "**Python 3.13** under *Advanced settings*."
    )
    st.stop()

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
AGENTS_YAML = ROOT / "src" / "startup_crew" / "config" / "agents.yaml"
PROJECTS_DIR = ROOT / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)
PROFILES_PATH = ROOT / "llm_profiles.json"

load_dotenv(ENV_PATH)

# Streamlit Community Cloud (and any host without a writable .env): seed the
# environment from st.secrets. Local .env values, if present, win.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and not os.environ.get(_k):
            os.environ[_k] = _v
except Exception:
    pass

sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------
# LLM provider presets
# --------------------------------------------------------------------------
PROVIDER_PRESETS = {
    "OpenRouter — Nemotron 3 Ultra (free)": dict(
        model="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        key_label="OpenRouter API key",
        key_help="Get one at openrouter.ai/keys",
        needs_key=True,
        model_editable=False,
        base_url_editable=False,
    ),
    "OpenRouter — custom model": dict(
        model="openrouter/",
        base_url="https://openrouter.ai/api/v1",
        key_label="OpenRouter API key",
        key_help="Model slug from openrouter.ai/models, e.g. openrouter/anthropic/claude-sonnet-4.5",
        needs_key=True,
        model_editable=True,
        base_url_editable=False,
    ),
    "NVIDIA NIM": dict(
        model="nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        key_label="NVIDIA API key (nvapi-...)",
        key_help="build.nvidia.com. If you use OpenCode, run connect_opencode.py "
                 "to import the key and model it already uses.",
        needs_key=True,
        model_editable=True,
        base_url_editable=False,
    ),
    "OpenAI": dict(
        model="gpt-4o-mini",
        base_url="",
        key_label="OpenAI API key",
        key_help="Get one at platform.openai.com/api-keys",
        needs_key=True,
        model_editable=True,
        base_url_editable=False,
    ),
    "Anthropic (Claude)": dict(
        model="claude-sonnet-4-5-20250929",
        base_url="",
        key_label="Anthropic API key",
        key_help="Get one at console.anthropic.com",
        needs_key=True,
        model_editable=True,
        base_url_editable=False,
    ),
    "Google (Gemini)": dict(
        model="gemini/gemini-2.0-flash",
        base_url="",
        key_label="Google AI Studio API key",
        key_help="Get one at aistudio.google.com/apikey",
        needs_key=True,
        model_editable=True,
        base_url_editable=False,
    ),
    "Local (Ollama)": dict(
        model="ollama/llama3.1",
        base_url="http://localhost:11434",
        key_label=None,
        key_help="Requires Ollama running locally (ollama.com) with the model pulled.",
        needs_key=False,
        model_editable=True,
        base_url_editable=True,
    ),
    "Custom OpenAI-compatible endpoint": dict(
        model="",
        base_url="",
        key_label="API key",
        key_help="Any OpenAI-compatible endpoint (vLLM, LM Studio, NVIDIA NIM, etc.)",
        needs_key=True,
        model_editable=True,
        base_url_editable=True,
    ),
}

st.set_page_config(page_title="Startup Crew", page_icon="🧑‍💼", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Inter, sans-serif; }

    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
    section[data-testid="stSidebar"] .stButton button {
        text-align: left; justify-content: flex-start; border: none; background: transparent;
        color: #D8D3CD; font-weight: 400; border-radius: 8px; padding: 0.4rem 0.6rem;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.06); color: #fff; }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: rgba(204,120,92,0.16); color: #F0B39C; font-weight: 500;
    }

    .screw-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .screw-badge {
        display:inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
        background: rgba(204,120,92,0.16); color:#E8A488; font-size:0.76rem;
        border: 1px solid rgba(204,120,92,0.35);
    }
    .screw-role { font-weight:600; font-size:1.05rem; color:#F3EFEA; }
    .screw-muted { color:#93897F; font-size:0.83rem; }
    .screw-brand { display:flex; align-items:center; gap:0.5rem; font-size:1.25rem; font-weight:600; color:#F3EFEA; }
    .screw-sidebar-section { color:#7F766D; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;
        margin: 0.9rem 0 0.2rem 0.3rem; }

    [data-testid="stChatMessage"] { border-radius: 16px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# 3D team constellation (Three.js) — shown on the empty-chat welcome screen.
# A literal (non-f) string so none of the JS/CSS braces need escaping; the
# single "__PAYLOAD__" token is swapped for real agent data at render time.
# --------------------------------------------------------------------------
_CONSTELLATION_TEMPLATE = """
<style>html, body { margin:0; padding:0; height:100%; overflow:hidden; }</style>
<div id="stage" style="width:100%; height:100%; position:relative; cursor:grab;"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function () {
  const payload = __PAYLOAD__;
  const stage = document.getElementById('stage');

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, stage.clientWidth / stage.clientHeight, 0.1, 100);
  camera.position.set(0, 0.6, 7.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  stage.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.minDistance = 4.5;
  controls.maxDistance = 10;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.1;
  controls.addEventListener('start', () => { stage.style.cursor = 'grabbing'; });
  controls.addEventListener('end', () => { stage.style.cursor = 'grab'; });

  scene.add(new THREE.AmbientLight(0xfff2ea, 0.55));
  const key = new THREE.PointLight(0xffcdb0, 1.4, 18);
  key.position.set(3, 4, 5);
  scene.add(key);
  const core = new THREE.PointLight(0xcc785c, 2.2, 8);
  scene.add(core);

  const managerGeo = new THREE.IcosahedronGeometry(0.85, 1);
  const managerMat = new THREE.MeshStandardMaterial({
    color: 0xcc785c, emissive: 0x7a3d29, emissiveIntensity: 0.6, metalness: 0.35, roughness: 0.35,
  });
  const managerMesh = new THREE.Mesh(managerGeo, managerMat);
  scene.add(managerMesh);

  const wire = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.02, 1),
    new THREE.MeshBasicMaterial({ color: 0xf0b39c, wireframe: true, transparent: true, opacity: 0.25 })
  );
  managerMesh.add(wire);

  const specialists = payload.specialists;
  const n = Math.max(specialists.length, 1);
  const nodeGroup = new THREE.Group();
  scene.add(nodeGroup);
  const nodes = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  const radius = 2.75;

  for (let i = 0; i < specialists.length; i++) {
    const y = 1 - (i / Math.max(n - 1, 1)) * 2;
    const r = Math.sqrt(Math.max(1 - y * y, 0));
    const theta = golden * i;
    const pos = new THREE.Vector3(Math.cos(theta) * r, y, Math.sin(theta) * r).multiplyScalar(radius);

    const geo = new THREE.SphereGeometry(0.42, 24, 24);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xe8a488, emissive: 0x3a2018, emissiveIntensity: 0.3, metalness: 0.2, roughness: 0.55,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    mesh.userData.role = specialists[i];
    mesh.userData.baseScale = 1;
    nodeGroup.add(mesh);
    nodes.push(mesh);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), pos]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xcc785c, transparent: true, opacity: 0.28 });
    nodeGroup.add(new THREE.Line(lineGeo, lineMat));
  }

  const label = document.createElement('div');
  label.style.cssText =
    'position:absolute; transform:translate(-50%,-140%); pointer-events:none; ' +
    'background:rgba(25,24,23,0.92); color:#F0B39C; border:1px solid rgba(204,120,92,0.5); ' +
    'padding:4px 10px; border-radius:999px; font:500 13px -apple-system,Segoe UI,sans-serif; ' +
    'white-space:nowrap; opacity:0; transition:opacity 0.15s;';
  stage.appendChild(label);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let hovered = null;

  stage.addEventListener('mousemove', (e) => {
    const rect = stage.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  });
  stage.addEventListener('mouseleave', () => {
    mouse.x = -999;
    hovered = null;
    label.style.opacity = 0;
  });

  const managerLabel = payload.manager;
  managerMesh.userData.role = managerLabel;

  function updateHover() {
    raycaster.setFromCamera(mouse, camera);
    const hit = raycaster.intersectObjects([managerMesh, ...nodes]);
    if (hit.length) {
      hovered = hit[0].object;
    } else {
      hovered = null;
      label.style.opacity = 0;
    }
  }

  function positionLabel(mesh) {
    const v = mesh.position.clone().project(camera);
    const x = (v.x * 0.5 + 0.5) * stage.clientWidth;
    const y = (-v.y * 0.5 + 0.5) * stage.clientHeight;
    label.style.left = x + 'px';
    label.style.top = y + 'px';
    label.textContent = mesh.userData.role;
    label.style.opacity = 1;
  }

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    managerMesh.scale.setScalar(1 + Math.sin(t * 1.4) * 0.04);
    wire.rotation.y += 0.004;
    nodes.forEach((m, i) => {
      const target = hovered === m ? 1.35 : 1;
      m.scale.setScalar(m.scale.x + (target - m.scale.x) * 0.15);
    });
    if (hovered) positionLabel(hovered);
    controls.update();
    updateHover();
    renderer.render(scene, camera);
  }
  animate();

  new ResizeObserver(() => {
    if (!stage.clientWidth || !stage.clientHeight) return;
    camera.aspect = stage.clientWidth / stage.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(stage.clientWidth, stage.clientHeight);
  }).observe(stage);
})();
</script>
"""


def render_team_constellation(agents_data: dict, height: int = 430) -> None:
    manager_role = next(
        (v.get("role", k).strip() for k, v in agents_data.items() if v.get("manager")), "Manager"
    )
    specialists = [v.get("role", k).strip() for k, v in agents_data.items() if not v.get("manager")]
    payload = json.dumps({"manager": manager_role, "specialists": specialists})
    components.html(_CONSTELLATION_TEMPLATE.replace("__PAYLOAD__", payload), height=height, scrolling=False)


# --------------------------------------------------------------------------
# Agents config helpers
# --------------------------------------------------------------------------

def load_agents() -> dict:
    return yaml.safe_load(AGENTS_YAML.read_text(encoding="utf-8"))


def save_agents(data: dict) -> None:
    header = (
        "# =============================================================================\n"
        "# STARTUP CREW — AGENT ROSTER (edited via the web UI's Team tab)\n"
        "# =============================================================================\n\n"
    )
    AGENTS_YAML.write_text(
        header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# LLM config helpers
# --------------------------------------------------------------------------

def current_llm_config() -> dict:
    import os

    return dict(
        label=os.environ.get("LLM_PROVIDER_LABEL", ""),
        model=os.environ.get("LLM_MODEL", ""),
        base_url=os.environ.get("LLM_BASE_URL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=os.environ.get("LLM_TEMPERATURE", "0.4"),
    )


def activate_llm_config(label: str, model: str, base_url: str, api_key: str, temperature: float) -> None:
    import os

    pairs = {
        "LLM_PROVIDER_LABEL": label,
        "LLM_MODEL": model,
        "LLM_BASE_URL": base_url or "",
        "LLM_API_KEY": api_key or "",
        "LLM_TEMPERATURE": str(temperature),
    }
    for k, v in pairs.items():
        set_key(str(ENV_PATH), k, v)
        os.environ[k] = v


def test_llm(model: str, base_url: str, api_key: str) -> tuple[bool, str]:
    import litellm

    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16,
            # Same reasoning-model problem as in crew.py's get_llm(): a short
            # timeout makes a working model look broken.
            timeout=float(os.environ.get("LLM_TIMEOUT", "900")),
        )
        if base_url:
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        resp = litellm.completion(**kwargs)
        text = resp["choices"][0]["message"]["content"]
        return True, f"Model responded: {text!r}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# --------------------------------------------------------------------------
# LLM profile chain — an ordered list of (label, model, base_url, api_key).
# If the first one is overloaded/fails, the crew automatically retries the
# whole run with the next one, and so on. Lets you register several keys for
# the same free model, or mix providers, for resilience.
# --------------------------------------------------------------------------

def _profiles_from_env() -> list[dict]:
    """Seed the whole fallback chain from an LLM_PROFILES env var / secret.

    Hosted deploys (Streamlit Cloud) start with no llm_profiles.json, since
    that file holds API keys and is gitignored. Setting LLM_PROFILES to a
    JSON array of {label, model, base_url, api_key, temperature} objects
    recreates the full chain on first run. Ignored when the file exists.
    """
    raw = os.environ.get("LLM_PROFILES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("model"):
            continue
        out.append({
            "id": item.get("id") or uuid.uuid4().hex[:8],
            "label": item.get("label") or "Profile",
            "model": item["model"],
            "base_url": item.get("base_url", ""),
            "api_key": item.get("api_key", ""),
            "temperature": item.get("temperature", 0.4),
        })
    return out


def load_profiles() -> list[dict]:
    if not PROFILES_PATH.exists():
        seeded = _profiles_from_env()
        if seeded:
            save_profiles(seeded)
            return seeded
        cfg = current_llm_config()
        seed = (
            [{
                "id": uuid.uuid4().hex[:8],
                "label": cfg["label"] or "Primary",
                "model": cfg["model"],
                "base_url": cfg["base_url"],
                "api_key": cfg["api_key"],
                "temperature": cfg["temperature"],
            }]
            if cfg["model"]
            else []
        )
        save_profiles(seed)
        return seed
    try:
        return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_profiles(profiles: list[dict]) -> None:
    PROFILES_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def _mark_last_good(profile: dict) -> None:
    """Keep .env in sync with whichever profile last actually answered, so
    the CLI (which only knows the single-config scheme) stays consistent."""
    activate_llm_config(
        profile["label"], profile["model"], profile.get("base_url", ""),
        profile.get("api_key", ""), float(profile.get("temperature", 0.4) or 0.4),
    )


def run_crew_with_fallback(objective: str, history: str, profiles: list[dict]):
    """Try each profile in order; on failure, move to the next. Raises with
    every attempt's error if all of them fail."""
    import os

    from startup_crew.crew import StartupCrew  # lazy import: env vars set below must take effect

    errors = []
    for prof in profiles:
        os.environ["LLM_MODEL"] = prof["model"]
        os.environ["LLM_API_KEY"] = prof.get("api_key") or ""
        os.environ["LLM_BASE_URL"] = prof.get("base_url") or ""
        os.environ["LLM_TEMPERATURE"] = str(prof.get("temperature", 0.4) or 0.4)
        try:
            result = StartupCrew().crew().kickoff(inputs={"objective": objective, "history": history})
            _mark_last_good(prof)
            return result, prof["label"], errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{prof['label']}: {exc}")
            continue
    raise RuntimeError(
        "Every configured model failed:\n" + "\n".join(f"- {e}" for e in errors)
    )


# --------------------------------------------------------------------------
# Project (chat thread) storage — every turn, and its full agent log, is
# written to disk immediately so nothing is lost on refresh/restart.
# --------------------------------------------------------------------------

def _project_path(pid: str) -> Path:
    return PROJECTS_DIR / f"{pid}.json"


def list_projects() -> list[dict]:
    projects = []
    for p in PROJECTS_DIR.glob("*.json"):
        try:
            projects.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return projects


def load_project(pid: str) -> dict | None:
    path = _project_path(pid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_project(proj: dict) -> None:
    proj["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _project_path(proj["id"]).write_text(json.dumps(proj, indent=2), encoding="utf-8")


def create_project(name: str = "New chat") -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    proj = {"id": uuid.uuid4().hex[:12], "name": name, "created_at": now, "updated_at": now, "turns": []}
    save_project(proj)
    return proj


def delete_project(pid: str) -> None:
    _project_path(pid).unlink(missing_ok=True)


def build_history(turns: list[dict], max_turns: int = 6, max_chars: int = 900) -> str:
    if not turns:
        return ""
    parts = []
    for t in turns[-max_turns:]:
        result = t.get("result", "")
        if len(result) > max_chars:
            result = result[:max_chars] + " …[truncated]"
        parts.append(f"Founder: {t['objective']}\nCrew: {result}")
    return "\n---\n".join(parts)


# --------------------------------------------------------------------------
# State bootstrap
# --------------------------------------------------------------------------

if "view" not in st.session_state:
    st.session_state.view = "chat"
if "active_project" not in st.session_state:
    st.session_state.active_project = None

_projects = list_projects()
if st.session_state.active_project and not any(p["id"] == st.session_state.active_project for p in _projects):
    st.session_state.active_project = None
if st.session_state.active_project is None:
    if _projects:
        st.session_state.active_project = _projects[0]["id"]
    else:
        _new = create_project("New chat")
        st.session_state.active_project = _new["id"]
        _projects = list_projects()

profiles = load_profiles()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<div class='screw-brand'>🧑‍💼 Startup Crew</div>", unsafe_allow_html=True)
    if profiles:
        badge = profiles[0]["label"] + (f" +{len(profiles) - 1} fallback" if len(profiles) > 1 else "")
    else:
        badge = "no model set"
    st.markdown(f"<span class='screw-badge'>{badge}</span>", unsafe_allow_html=True)

    st.markdown("<div class='screw-sidebar-section'>Chats</div>", unsafe_allow_html=True)
    if st.button("✚ New chat", use_container_width=True):
        _new = create_project("New chat")
        st.session_state.active_project = _new["id"]
        st.session_state.view = "chat"
        st.rerun()
    if st.button(
        "🗂 All projects", use_container_width=True,
        type=("primary" if st.session_state.view == "all_projects" else "secondary"),
    ):
        st.session_state.view = "all_projects"
        st.rerun()

    for p in _projects[:8]:
        row_l, row_r = st.columns([5, 1])
        is_active = p["id"] == st.session_state.active_project and st.session_state.view == "chat"
        if row_l.button(
            p["name"] or "New chat",
            key=f"sel_{p['id']}",
            use_container_width=True,
            type=("primary" if is_active else "secondary"),
        ):
            st.session_state.active_project = p["id"]
            st.session_state.view = "chat"
            st.rerun()
        with row_r.popover("⋮"):
            new_name = st.text_input("Rename", value=p["name"], key=f"rn_{p['id']}", label_visibility="collapsed")
            if st.button("Save name", key=f"rnsave_{p['id']}", use_container_width=True):
                proj = load_project(p["id"])
                proj["name"] = new_name.strip() or "New chat"
                save_project(proj)
                st.rerun()
            st.divider()
            confirm = st.checkbox("Confirm delete", key=f"delconf_{p['id']}")
            if st.button("🗑 Delete chat", key=f"del_{p['id']}", disabled=not confirm, use_container_width=True):
                delete_project(p["id"])
                if st.session_state.active_project == p["id"]:
                    st.session_state.active_project = None
                st.rerun()

    if len(_projects) > 8:
        st.caption(f"+{len(_projects) - 8} more — see 🗂 All projects")

    st.markdown("<div class='screw-sidebar-section'>Configure</div>", unsafe_allow_html=True)
    if st.button("🧑‍🤝‍🧑 Team", use_container_width=True, type=("primary" if st.session_state.view == "team" else "secondary")):
        st.session_state.view = "team"
        st.rerun()
    if st.button("🔌 LLM & Model", use_container_width=True, type=("primary" if st.session_state.view == "model" else "secondary")):
        st.session_state.view = "model"
        st.rerun()
    if st.button("ℹ️ About", use_container_width=True, type=("primary" if st.session_state.view == "about" else "secondary")):
        st.session_state.view = "about"
        st.rerun()

# --------------------------------------------------------------------------
# Main: Chat (project)
# --------------------------------------------------------------------------

if st.session_state.view == "chat":
    proj = load_project(st.session_state.active_project)

    st.markdown(f"## {proj['name']}")

    if not proj["turns"]:
        render_team_constellation(load_agents())
        st.caption(
            "Drag to rotate, hover a node to see who's on the team. The Chief of Staff "
            "(center) delegates your message to whichever specialists are relevant, "
            "resolves conflicts between them, and answers as one team. This chat "
            "remembers everything said in it so far."
        )

    for turn in proj["turns"]:
        with st.chat_message("user"):
            st.markdown(turn["objective"])
        with st.chat_message("assistant", avatar="🧑‍💼"):
            st.markdown(turn["result"])
            if turn.get("fallback_errors"):
                st.caption(
                    f"⚠️ {len(turn['fallback_errors'])} model(s) in the chain failed before "
                    f"'{turn.get('model_label', '')}' answered — see log for details."
                )
            if turn.get("log") or turn.get("fallback_errors"):
                with st.expander("Agent activity log"):
                    if turn.get("fallback_errors"):
                        st.markdown("**Fallback attempts:**\n" + "\n".join(f"- {e}" for e in turn["fallback_errors"]))
                    st.code(turn.get("log", ""), language=None)
            st.caption(f"{turn.get('model_label', '')} · {turn.get('timestamp', '')}")

    err_key = f"error_{proj['id']}"
    if st.session_state.get(err_key):
        st.error(f"The crew hit an error: {st.session_state[err_key]}")
        st.caption(
            "If this mentions 'overloaded' or a timeout, the endpoint is temporarily busy "
            "(common on free tiers) — just try again, or switch models in 🔌 LLM & Model."
        )

    if not profiles:
        st.warning("No LLM is configured yet — go to **🔌 LLM & Model** first.")

    msg = st.chat_input("Message your crew...")
    if msg:
        history_str = build_history(proj["turns"])
        log_buffer = io.StringIO()
        n = len(profiles)
        spinner_text = (
            "Crew is working — delegating to specialists and synthesizing a recommendation..."
            if n <= 1
            else f"Crew is working (up to {n} models in your fallback chain will be tried if one fails)..."
        )
        with st.spinner(spinner_text):
            try:
                with redirect_stdout(log_buffer):
                    result, used_label, fallback_errors = run_crew_with_fallback(msg, history_str, profiles)
                turn = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "objective": msg,
                    "result": result.raw,
                    "log": log_buffer.getvalue(),
                    "model_label": used_label,
                    "fallback_errors": fallback_errors,
                }
                proj["turns"].append(turn)
                if proj["name"] == "New chat" and len(proj["turns"]) == 1:
                    proj["name"] = msg[:48] + ("…" if len(msg) > 48 else "")
                save_project(proj)
                st.session_state.pop(err_key, None)
            except Exception as exc:  # noqa: BLE001
                st.session_state[err_key] = str(exc)
        st.rerun()

# --------------------------------------------------------------------------
# Main: All Projects
# --------------------------------------------------------------------------

elif st.session_state.view == "all_projects":
    st.header("All projects")
    st.caption(f"{len(_projects)} chat project(s) total, most recently updated first.")

    query = st.text_input("Search by name or message content", key="all_projects_search")
    q = query.strip().lower()

    for p in _projects:
        if q and q not in p["name"].lower() and not any(q in t.get("objective", "").lower() for t in p["turns"]):
            continue
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 2, 2])
            c1.markdown(f"**{p['name']}**")
            c1.caption(f"{len(p['turns'])} message(s)")
            c2.caption(f"Created\n{p.get('created_at', '')[:16].replace('T', ' ')}")
            c3.caption(f"Updated\n{p.get('updated_at', '')[:16].replace('T', ' ')}")
            b1, b2 = st.columns(2)
            if b1.button("Open", key=f"open_{p['id']}", use_container_width=True, type="primary"):
                st.session_state.active_project = p["id"]
                st.session_state.view = "chat"
                st.rerun()
            with b2.popover("🗑 Delete", use_container_width=True):
                st.caption(f"Permanently delete '{p['name']}'? This can't be undone.")
                if st.button("Confirm delete", key=f"ap_del_{p['id']}"):
                    delete_project(p["id"])
                    if st.session_state.active_project == p["id"]:
                        st.session_state.active_project = None
                    st.rerun()

    if not _projects:
        st.info("No projects yet — start one with **✚ New chat**.")

# --------------------------------------------------------------------------
# Main: Team
# --------------------------------------------------------------------------

elif st.session_state.view == "team":
    st.header("Your team")
    st.caption(
        "Add members, edit anyone's role/specialization/personality, delete anyone, "
        "or reassign who manages the team — all of it takes effect the next time "
        "you send a message. No code editing, ever."
    )

    agents_data = load_agents()
    keys = list(agents_data.keys())
    current_manager = next((k for k, v in agents_data.items() if v.get("manager")), keys[0])

    manager_choice = st.selectbox(
        "Manager — delegates work to everyone else and writes the final answer",
        keys,
        index=keys.index(current_manager),
        format_func=lambda k: agents_data[k].get("role", k).strip(),
    )

    st.divider()

    edited = {}
    delete_keys = set()

    for key, agent_def in agents_data.items():
        is_manager = key == manager_choice
        with st.container(border=True):
            top_l, top_r = st.columns([5, 1])
            top_l.markdown(
                f"<div class='screw-role'>{agent_def.get('role', key).strip()}</div>", unsafe_allow_html=True
            )
            if is_manager:
                top_r.markdown("<span class='screw-badge'>Manager</span>", unsafe_allow_html=True)
            role = st.text_input("Role / specialization", value=agent_def.get("role", "").strip(), key=f"{key}_role")
            goal = st.text_area(
                "Goal — what they're responsible for delivering",
                value=agent_def.get("goal", "").strip(),
                key=f"{key}_goal",
                height=90,
            )
            backstory = st.text_area(
                "Backstory / personality — shapes tone, priorities, and judgment calls",
                value=agent_def.get("backstory", "").strip(),
                key=f"{key}_backstory",
                height=110,
            )
            edited[key] = {"role": role, "goal": goal, "backstory": backstory, "manager": is_manager}

            if len(agents_data) > 1 and st.button("🗑 Remove from team", key=f"{key}_delete"):
                delete_keys.add(key)

    if st.button("💾 Save team", type="primary"):
        for k in delete_keys:
            edited.pop(k, None)
        if delete_keys and manager_choice in delete_keys:
            remaining = list(edited.keys())
            edited[remaining[0]]["manager"] = True
        save_agents(edited)
        st.success("Saved to config/agents.yaml." + (f" Removed: {', '.join(delete_keys)}." if delete_keys else ""))
        st.rerun()

    st.divider()
    with st.expander("➕ Add a new team member"):
        new_key = st.text_input("Internal key (snake_case, unique)", placeholder="e.g. legal_advisor", key="new_key")
        new_role = st.text_input("Role / specialization", placeholder="e.g. Legal & Compliance Advisor", key="new_role")
        new_goal = st.text_area("Goal — what they're responsible for delivering", key="new_goal", height=80)
        new_backstory = st.text_area(
            "Backstory / personality — shapes tone, priorities, and judgment calls",
            key="new_backstory",
            height=100,
        )
        if st.button("Add to team"):
            if not new_key or not new_role or not new_goal or not new_backstory:
                st.error("Key, role, goal, and backstory are all required.")
            elif new_key in agents_data:
                st.error(f"'{new_key}' already exists — pick a unique key.")
            else:
                agents_data[new_key] = {"role": new_role, "goal": new_goal, "backstory": new_backstory, "manager": False}
                save_agents(agents_data)
                st.success(f"Added '{new_role}' to the team.")
                st.rerun()

# --------------------------------------------------------------------------
# Main: LLM & Model
# --------------------------------------------------------------------------

elif st.session_state.view == "model":
    st.header("LLM & model chain")
    st.caption(
        "Listed in priority order. If the first model/key fails or is overloaded, the crew "
        "automatically retries the *whole run* with the next one — no manual intervention. "
        "Register several keys for the same free model to dodge rate limits, or mix providers "
        "for resilience."
    )

    if not profiles:
        st.info("No models configured yet — add one below.")

    for idx, prof in enumerate(profiles):
        with st.container(border=True):
            top_l, top_r = st.columns([5, 3])
            top_l.markdown(
                f"<span class='screw-badge'>#{idx + 1}</span> "
                f"<span class='screw-role'>{prof['label']}</span><br>"
                f"<span class='screw-muted'>{prof['model']}</span>",
                unsafe_allow_html=True,
            )
            b1, b2, b3, b4, b5 = top_r.columns(5)
            if b1.button("↑", key=f"up_{prof['id']}", disabled=idx == 0, help="Move up (higher priority)"):
                profiles[idx - 1], profiles[idx] = profiles[idx], profiles[idx - 1]
                save_profiles(profiles)
                st.rerun()
            if b2.button("↓", key=f"down_{prof['id']}", disabled=idx == len(profiles) - 1, help="Move down"):
                profiles[idx + 1], profiles[idx] = profiles[idx], profiles[idx + 1]
                save_profiles(profiles)
                st.rerun()
            with b3.popover("🔎", help="Test connection"):
                if st.button("Run test", key=f"testrun_{prof['id']}"):
                    with st.spinner("Calling the model..."):
                        ok, msgres = test_llm(prof["model"], prof.get("base_url", ""), prof.get("api_key", ""))
                    (st.success if ok else st.error)(msgres)
            with b4.popover("✎", help="Edit"):
                e_label = st.text_input("Label", value=prof["label"], key=f"e_label_{prof['id']}")
                e_model = st.text_input("Model", value=prof["model"], key=f"e_model_{prof['id']}")
                e_base = st.text_input("Base URL", value=prof.get("base_url", ""), key=f"e_base_{prof['id']}")
                e_key = st.text_input(
                    "API key", value=prof.get("api_key", ""), type="password", key=f"e_key_{prof['id']}"
                )
                e_temp = st.slider(
                    "Temperature", 0.0, 1.0, float(prof.get("temperature", 0.4) or 0.4), 0.1,
                    key=f"e_temp_{prof['id']}",
                )
                if st.button("Save changes", key=f"e_save_{prof['id']}", type="primary"):
                    profiles[idx] = {
                        **prof, "label": e_label, "model": e_model, "base_url": e_base,
                        "api_key": e_key, "temperature": e_temp,
                    }
                    save_profiles(profiles)
                    st.rerun()
            if b5.button("🗑", key=f"rm_{prof['id']}", help="Remove from chain"):
                profiles.pop(idx)
                save_profiles(profiles)
                st.rerun()

    st.divider()
    with st.expander("➕ Add a model/key to the chain", expanded=not profiles):
        labels = list(PROVIDER_PRESETS.keys())
        choice = st.selectbox("Provider", labels, key="add_provider_choice")
        preset = PROVIDER_PRESETS[choice]

        new_label = st.text_input("Label (shown in the chain)", value=choice, key="add_label")
        new_model = st.text_input(
            "Model", value=preset["model"], disabled=not preset["model_editable"], help=preset["key_help"],
            key="add_model",
        )
        new_base = st.text_input(
            "Base URL", value=preset["base_url"], disabled=not preset["base_url_editable"], key="add_base"
        )
        new_key = ""
        if preset["needs_key"]:
            new_key = st.text_input(preset["key_label"], type="password", help=preset["key_help"], key="add_key")
        else:
            st.info(preset["key_help"])
        new_temp = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1, key="add_temp")

        c1, c2 = st.columns(2)
        if c1.button("🔎 Test before adding", key="add_test"):
            with st.spinner("Calling the model..."):
                ok, msgres = test_llm(new_model, new_base, new_key)
            (st.success if ok else st.error)(msgres)
        if c2.button("➕ Add to chain", type="primary", key="add_confirm"):
            if preset["needs_key"] and not new_key:
                st.error(f"{preset['key_label']} is required for this provider.")
            else:
                profiles.append({
                    "id": uuid.uuid4().hex[:8], "label": new_label, "model": new_model,
                    "base_url": new_base, "api_key": new_key, "temperature": new_temp,
                })
                save_profiles(profiles)
                st.success(f"Added '{new_label}' to the chain.")
                st.rerun()

# --------------------------------------------------------------------------
# Main: About
# --------------------------------------------------------------------------

else:
    st.header("About this crew")
    st.markdown(
        f"""
This is a self-hosted [CrewAI](https://github.com/crewAIInc/crewAI) team running
entirely on your PC — nothing leaves your machine except the LLM API calls
themselves. Three things you can change without touching code:

- **Who's on the team** — add/edit/remove/reassign manager in **🧑‍🤝‍🧑 Team**.
- **What they're asked to do** — edit `src/startup_crew/config/tasks.yaml`.
- **Which AI powers them** — manage a priority-ordered chain of models/keys in
  **🔌 LLM & Model**. If the first one fails or is overloaded, the crew
  automatically retries the whole run with the next one in the chain.

**Chats are projects.** Each one in the sidebar (or browse all of them in
**🗂 All projects**) remembers its own conversation — the crew sees prior
messages/answers in that chat as memory, so follow-ups build on earlier
context instead of starting cold. Every message, answer, and full agent
activity log is saved to `projects/<id>.json` immediately, so nothing is lost
on refresh or restart.

Project folder: `{ROOT}`

See `README.md` in that folder for the CLI equivalent and deeper customization
(adding tools like web search, changing the task pipeline, etc).
"""
    )
