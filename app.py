#!/usr/bin/env python
# coding: utf-8
import os, re, json, base64, requests
from io import BytesIO
from PIL import Image
import streamlit as st
import validators
from dotenv import load_dotenv
from openai import OpenAI
# ============== APP SETUP ==============
st.set_page_config(page_title="VoyllaGPT • Vision + Prompt (Chat Mode)", layout="wide")
st.title("✨ VoyllaGPT — Vision + Prompt (Chat History)")
st.caption("Analyze images → standardised traits · Generate manufacturable jewelry concepts (DALL·E 3)")
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Please set your OPENAI_API_KEY in your .env file")
    st.stop()
client = OpenAI()
# ===== SIZE / QUALITY CONFIG =====
MAX_INPUT_MB   = 5      # reject larger files from URL/upload
PROCESS_MAX_PX = 768    # image sent to GPT-4o vision
PREVIEW_MAX_PX = 400    # preview width/height in UI
DOWNLOAD_MAX_PX= 1024   # cap "full" image for download
TIMEOUT_SEC    = 15
JPEG_QUALITY   = 90
# ------------- CONTROLLED VOCAB -------------
ALLOWED = {
    "Design Style": ["Contemporary","Traditional/Ethnic","Vintage","Minimalist","Classic","Glamorous","Tribal","Geometric","Bohemian","Statement","Fusion","Art Deco"],
    "Form": ["Stud","Hoop","Drop/Dangle","Chandbali (Crescent)","Jhumka","Ear Cuff","Teardrop","Triangle","Semicircle","Crescent","Round","Oval","Square","Rectangle","Heart","Floral","Band","Link","Pendant","Bangle","Cuff Bracelet","Geometric"],
    "Metal Color": ["Yellow Gold","Rose Gold","Silver","Antique Silver","Antique Gold","Oxidized Black"],
    "Craft Style": ["Handcrafted","Casting","Enamel/Meenakari","Kundan","Filigree","Jaali Work","Stone-Set","Engraved/Etched","Oxidized Finish","Precision Cut"],
    "Central Stone": ["None","Cubic Zirconia","Pearl","Lab-Grown Diamond","Diamond","Glass/Kundan","Enamel","Green Stone","Red Stone","Blue Stone","Multi-stone","Clear Stone","Gemstone"],
    "Surrounding Layout": ["Plain","Halo","Pavé Accents","Cluster","Enamel Panel","Beaded Fringe","Kundan Border","Accented"],
    "Stone Setting": ["None","Prong","Bezel","Pavé","Channel","Kundan","Flush","Tension","Adhesive","Mixed"],
    "Style Motif": ["Geometric","Floral","Tribal","Animal","Heritage","Minimalist","Heart","Religious","Modern"],
}
KW = {
    "Design Style": [(r"(contemporary|modern)","Contemporary"),(r"(traditional|ethnic|heritage|rajwada|temple)","Traditional/Ethnic"),
                     (r"(vintage|retro)","Vintage"),(r"minimal","Minimalist"),(r"classic","Classic"),(r"glam","Glamorous"),
                     (r"tribal","Tribal"),(r"geometric|geo","Geometric"),(r"boho|bohemian","Bohemian"),(r"statement|bold","Statement"),
                     (r"art\s*deco","Art Deco"),(r"fusion|mix","Fusion")],
    "Form": [(r"\bear\s*cuff\b","Ear Cuff"),(r"\bjhumk?a\b","Jhumka"),
             (r"chandbali|crescent","Chandbali (Crescent)"),(r"\bstud\b","Stud"),
             (r"huggie|hoop","Hoop"),(r"drop|dangle","Drop/Dangle"),(r"teardrop|pear","Teardrop"),
             (r"triangle|triangular","Triangle"),(r"semi.?circle|half.?moon","Semicircle"),(r"\bcrescent\b","Crescent"),
             (r"round|circle","Round"),(r"\boval\b","Oval"),(r"square","Square"),(r"rectang","Rectangle"),
             (r"heart","Heart"),(r"flor|flower|leaf","Floral"),(r"\bband\b","Band"),(r"\blink\b","Link"),
             (r"pendant","Pendant"),(r"bangle","Bangle"),(r"cuff bracelet","Cuff Bracelet"),(r"geo","Geometric")],
    "Metal Color": [(r"rose","Rose Gold"),(r"yellow|gold\b|gold[- ]tone","Yellow Gold"),(r"antique.*silver","Antique Silver"),
                    (r"antique","Antique Gold"),(r"oxidiz|gunmetal|black","Oxidized Black"),(r"silver|white","Silver")],
    "Craft Style": [(r"hand.?made|hand.?crafted|hand work","Handcrafted"),(r"\bcasting?\b|\bcast\b","Casting"),
                    (r"meenakari|meena|enamel","Enamel/Meenakari"),(r"kundan","Kundan"),
                    (r"filigree|filgree","Filigree"),(r"ja+li|jaliya|jalli","Jaali Work"),
                    (r"stone.?set|studded|pav[eé]|prong|channel|bezel|micro.?pave","Stone-Set"),
                    (r"engrave|etched|etching","Engraved/Etched"),(r"oxidiz|blackened|antique finish","Oxidized Finish"),
                    (r"precision cut|machine cut|faceted|faceting","Precision Cut")],
    "Central Stone": [(r"\bnone\b|no stone|n/?a","None"),(r"\bcubic zirconia\b|\bcz\b|zircon","Cubic Zirconia"),
                      (r"\bpearl\b","Pearl"),(r"lab.*diamond","Lab-Grown Diamond"),(r"\bdiamond\b","Diamond"),
                      (r"glass|kundan","Glass/Kundan"),(r"enamel","Enamel"),(r"emerald|green stone","Green Stone"),
                      (r"ruby|red stone","Red Stone"),(r"sapphire|blue stone","Blue Stone"),(r"multi","Multi-stone"),
                      (r"clear|white stone","Clear Stone"),(r"stone","Gemstone")],
    "Surrounding Layout": [(r"\bplain|minimal","Plain"),(r"halo","Halo"),(r"pav[eé]","Pavé Accents"),
                           (r"cluster","Cluster"),(r"enamel","Enamel Panel"),(r"bead|fringe","Beaded Fringe"),
                           (r"kundan","Kundan Border"),(r"accent","Accented")],
    "Stone Setting": [(r"\bnone\b|no setting|n/?a","None"),(r"prong","Prong"),(r"bezel","Bezel"),
                      (r"pav[eé]","Pavé"),(r"channel","Channel"),(r"kundan","Kundan"),
                      (r"flush|gypsy","Flush"),(r"tension","Tension"),(r"adhesive|glue","Adhesive"),(r"set","Mixed")],
    "Style Motif": [(r"geometric|triangle|square|round|oval|jaali","Geometric"),(r"flor|petal|leaf|flower","Floral"),
                    (r"tribal","Tribal"),(r"animal|peacock|elephant|frog","Animal"),
                    (r"heritage|royal|rajwada|traditional|temple|sanskrit","Heritage"),
                    (r"minimal","Minimalist"),(r"heart","Heart"),(r"relig","Religious"),(r"modern|clean","Modern")]
}
def map_to_allowed(field, text):
    if not text: return None
    t = text.lower()
    for pattern, label in KW[field]:
        if re.search(pattern, t): return label
    for label in ALLOWED[field]:
        if label.lower() in t: return label
    return None
def map_craft_multi(text, max_items=2):
    if not text: return None
    t = text.lower(); hits=[]
    for pattern, label in KW["Craft Style"]:
        if re.search(pattern, t) and label not in hits:
            hits.append(label)
        if len(hits)>=max_items: break
    return " | ".join(hits[:max_items]) if hits else None
# ------------- IMAGE UTILS -------------
def _bytes_to_mb(n:int)->float: return round(n/(1024*1024),2)
def open_image_from_bytes(raw:bytes)->Image.Image:
    img = Image.open(BytesIO(raw))
    if img.mode not in ("RGB","RGBA"): img = img.convert("RGB")
    return img
def resize_max_edge(img:Image.Image, max_px:int)->Image.Image:
    copy=img.copy(); copy.thumbnail((max_px,max_px), Image.LANCZOS); return copy
def to_png_bytes(img:Image.Image)->bytes:
    buf=BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()
def to_jpeg_bytes(img:Image.Image, quality:int=JPEG_QUALITY)->bytes:
    buf=BytesIO()
    if img.mode=="RGBA":
        bg=Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[-1]); img=bg
    img.save(buf, format="JPEG", quality=quality, optimize=True); return buf.getvalue()
def image_to_base64(image: Image.Image) -> str:
    return base64.b64encode(to_png_bytes(image)).decode("utf-8")
def fetch_image_from_url_small(url:str)->Image.Image|None:
    try:
        r=requests.get(url, timeout=TIMEOUT_SEC, stream=True); r.raise_for_status()
        total=int(r.headers.get("Content-Length") or 0)
        if total and _bytes_to_mb(total)>MAX_INPUT_MB:
            st.error(f"URL image too large ({_bytes_to_mb(total)} MB > {MAX_INPUT_MB} MB)."); return None
        buf=BytesIO(); read=0; cap=MAX_INPUT_MB*1024*1024
        for chunk in r.iter_content(1024*64):
            if not chunk: break
            read+=len(chunk)
            if read>cap: st.error(f"URL image exceeded {MAX_INPUT_MB} MB limit."); return None
            buf.write(chunk)
        return open_image_from_bytes(buf.getvalue())
    except Exception as e:
        st.error(f"Failed to fetch image: {e}"); return None
def handle_uploaded_file_small(uploaded_file)->Image.Image|None:
    raw=uploaded_file.read(); size=_bytes_to_mb(len(raw))
    if size>MAX_INPUT_MB: st.error(f"Uploaded image too large ({size} MB > {MAX_INPUT_MB} MB)."); return None
    return open_image_from_bytes(raw)
# ------------- VISION (standardised) -------------
def analyze_image_standardised(image: Image.Image):
    base64_image = image_to_base64(image)
    system = (
        "You are a jewelry product expert. "
        "Return STRICT JSON with keys exactly: "
        '["Design Style","Form","Metal Color","Craft Style","Central Stone","Surrounding Layout","Stone Setting","Style Motif","detailed_summary","similar_prompt"]. '
        "For detailed_summary: Write 8-10 sentences describing all visible details including proportions, textures, finishing, patterns, and manufacturing aspects. "
        "For similar_prompt: Create a concise DALL-E prompt (2-3 sentences) that captures the key visual elements to recreate a similar piece. "
        "Values must be descriptive; use 'None' only when clearly no stone/setting."
    )
    user = (
        "Analyze the jewelry image and fill the JSON. Focus on:\n"
        "1. All visible design elements and proportions\n"
        "2. Surface treatments, textures, and finishes\n"
        "3. Construction details and joinery\n"
        "4. Color variations and material interactions\n"
        "5. Size relationships between components\n"
        "Create a detailed summary that captures manufacturing nuances and a similar_prompt for recreation."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":[
                {"type":"text","text":user},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=800  # Increased for detailed summary
    )
    raw = resp.choices[0].message.content.strip()
    try:
        raw_json = re.sub(r"^```(json)?|```$","",raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw_json)
    except Exception:
        data = {k:None for k in ["Design Style","Form","Metal Color","Craft Style","Central Stone","Surrounding Layout","Stone Setting","Style Motif","detailed_summary","similar_prompt"]}
        for k in data.keys():
            m=re.search(rf"{re.escape(k)}\s*[:\-]\s*(.+)", raw, flags=re.I)
            if m: data[k]=m.group(1).strip()
    std={}
    std["Design Style"]=map_to_allowed("Design Style", data.get("Design Style") or "") or "Contemporary"
    std["Form"]=map_to_allowed("Form", data.get("Form") or "") or "Geometric"
    std["Metal Color"]=map_to_allowed("Metal Color", data.get("Metal Color") or "") or "Silver"
    std["Craft Style"]=map_craft_multi(data.get("Craft Style") or data.get("detailed_summary") or "")  # may be None
    std["Central Stone"]=map_to_allowed("Central Stone", data.get("Central Stone") or "") or "None"
    std["Surrounding Layout"]=map_to_allowed("Surrounding Layout", data.get("Surrounding Layout") or "") or "Plain"
    std["Stone Setting"]=map_to_allowed("Stone Setting", data.get("Stone Setting") or ("None" if std["Central Stone"] in ["None",None] else "")) or "None"
    std["Style Motif"]=map_to_allowed("Style Motif", data.get("Style Motif") or "") or "Modern"
    std["detailed_summary"]=data.get("detailed_summary") or "Detailed design attributes extracted and standardised for manufacturing reference."
    std["similar_prompt"]=data.get("similar_prompt") or f"{std['Design Style']} {std['Form'].lower()} in {std['Metal Color'].lower()}, {std['Style Motif'].lower()} motif with {std['Central Stone'].lower()} stone"
    return std
# ------------- GENERATION (DALL·E 3) -------------
GEN_SYS_SAFETY = (
    "You are generating jewelry images for MANUFACTURING REFERENCE, not concept art. "
    "Output must be realistic and buildable by human artisans using common techniques. "
    "STRICT CONSTRAINTS:\n"
    "- Keep forms simple, clean geometry, clear negative space.\n"
    "- Max 2–3 layers of detailing; avoid dense filigree webs or micro-granulation.\n"
    "- Use standard feasible findings: prong/bezels/channel, jump rings, hooks, posts, clasps, chains.\n"
    "- No floating/overlapping parts, impossible interlocks, ultra-tiny beads or hairline wires.\n"
    "- Stones must be seatable; believable prongs/bezels.\n"
    "- Materials: Yellow/Rose Gold, Silver/Brass, Enamel/Meenakari, Kundan, Pearls, CZ, Diamonds.\n"
    "- Finishes: Smooth polished, matte, brushed, oxidized (blackened).\n"
    "- Show closures clearly; single product on neutral background (catalog style).\n"
    "- ABSOLUTELY NO TEXT/ANNOTATIONS/WATERMARKS/DIAGRAMS/RULERS."
)
def build_generation_prompt(user_text:str)->str:
    guidelines = (
        "RENDERING GUIDELINES:\n"
        "- View: top-down or 3/4 product shot; edges & thickness visible.\n"
        "- Composition: single piece centered; neutral backdrop; crisp lighting, realistic metal reflections.\n"
        "- Complexity cap: repeatable motifs only (floral, geometric, heritage, minimalist).\n"
        "- Clarity: emphasise manufacturable joins; avoid wire-like lattices & needle-thin details.\n"
        "- Quality: photo-style product render (not illustration)."
    )
    return f"{GEN_SYS_SAFETY}\n\nUSER BRIEF:\n{user_text}\n\n{guidelines}"
def dalle_generate_image(user_text:str, gen_size="512x512", preview_max_px=PREVIEW_MAX_PX, download_max_px=DOWNLOAD_MAX_PX):
    prompt = build_generation_prompt(user_text.strip())
    try:
        resp = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=gen_size,              # "512x512" / "768x768" / "1024x1024"
            n=1,
            quality="standard",         # or "hd"
            response_format="b64_json"
        )
        if not resp or not resp.data:
            return None, None, "No data returned."
        b64 = resp.data[0].b64_json
        img_full = open_image_from_bytes(base64.b64decode(b64))  # original gen (already small)
        img_preview = resize_max_edge(img_full, preview_max_px)
        img_download = resize_max_edge(img_full, download_max_px)
        return img_preview, img_download, None
    except Exception as e:
        return None, None, str(e)
# ------------- STATE (history persists) -------------
if "history" not in st.session_state:
    st.session_state.history = []  # items: {"type": "analysis"/"generation", ...}
# ------------- SIDEBAR ACTIONS -------------
with st.sidebar:
    st.header("🛠 Actions")
    st.markdown("**Analyze an Image**")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="upl")
    url = st.text_input("...or paste image URL")
    if st.button("Analyze", use_container_width=True):
        img = None
        if up is not None:
            img = handle_uploaded_file_small(up)
        elif url and validators.url(url) and url.lower().endswith((".jpg",".jpeg",".png",".webp")):
            img = fetch_image_from_url_small(url)
        if img is None:
            st.warning("Please provide a valid image (≤ 5 MB).")
        else:
            # prepare size variants
            img_for_vision = resize_max_edge(img, PROCESS_MAX_PX)
            img_preview    = resize_max_edge(img, PREVIEW_MAX_PX)
            img_download   = resize_max_edge(img, DOWNLOAD_MAX_PX)
            with st.spinner("🧠 Analyzing and standardising..."):
                attrs = analyze_image_standardised(img_for_vision)
            st.session_state.history.append({
                "type": "analysis",
                "image_bytes_preview": to_jpeg_bytes(img_preview),
                "image_bytes_full": to_png_bytes(img_download),
                "attributes": attrs
            })
    st.markdown("---")
    st.markdown("**Generate with DALL·E 3**")
    gen_text = st.text_area("Design prompt", placeholder="e.g. Minimalist oxidized silver hoops, prong-set CZ, geometric motif")
    size_choice = st.selectbox("Generation size", ["1024x1024"], index=0)
    if st.button("Generate Image", use_container_width=True):
        if not gen_text.strip():
            st.warning("Enter a prompt.")
        else:
            with st.spinner("✨ Generating manufacturable concept…"):
                pvw, dwn, err = dalle_generate_image(gen_text, gen_size=size_choice)
            if err:
                st.error(f"❌ DALL·E failed: {err}")
            elif pvw and dwn:
                st.session_state.history.append({
                    "type": "generation",
                    "prompt": gen_text.strip(),
                    "image_bytes_preview": to_jpeg_bytes(pvw),
                    "image_bytes_full": to_png_bytes(dwn)
                })
# ------------- HISTORY RENDER -------------
st.subheader("💬 History")
if not st.session_state.history:
    st.info("No items yet. Use the sidebar to analyze an image or generate one.")
else:
    for i, item in enumerate(st.session_state.history):
        if item["type"] == "analysis":
            with st.chat_message("assistant"):
                st.markdown("**Image Analysis (standardised traits)**")
                st.image(Image.open(BytesIO(item["image_bytes_preview"])), width=PREVIEW_MAX_PX)
                a = item["attributes"]
                rows = [
                    ("Design Style", a["Design Style"]),
                    ("Form", a["Form"]),
                    ("Metal Color", a["Metal Color"]),
                    ("Craft Style", a["Craft Style"] or "—"),
                    ("Central Stone", a["Central Stone"]),
                    ("Surrounding Layout", a["Surrounding Layout"]),
                    ("Stone Setting", a["Stone Setting"]),
                    ("Style Motif", a["Style Motif"]),
                ]
                st.table(rows)
                
                # Enhanced display for detailed summary and similar prompt
                st.markdown("**📋 Detailed Analysis**")
                st.markdown(f"{a['detailed_summary']}")
                
                st.markdown("**🎯 Similar Product Prompt**")
                st.code(a['similar_prompt'], language="text")
                
                # Add copy button for the similar prompt
                if st.button(f"📋 Copy Similar Prompt", key=f"copy_prompt_{i}"):
                    st.write("✅ Prompt copied! You can now paste it in the generation text area above.")
                    st.session_state[f"copied_prompt_{i}"] = a['similar_prompt']
                
                st.download_button(
                    "📥 Download analyzed PNG",
                    data=item["image_bytes_full"],
                    file_name=f"analyzed_{i+1}.png",
                    mime="image/png",
                    key=f"dl_an_{i}"
                )
        elif item["type"] == "generation":
            with st.chat_message("assistant"):
                st.markdown("**Generated Image (DALL·E 3)**")
                st.markdown(f"*Prompt:* {item['prompt']}")
                st.image(Image.open(BytesIO(item["image_bytes_preview"])), width=PREVIEW_MAX_PX)
                st.download_button(
                    "📥 Download PNG",
                    data=item["image_bytes_full"],
                    file_name=f"voylla_generated_{i+1}.png",
                    mime="image/png",
                    key=f"dl_gen_{i}"
                )
st.caption("State is persisted in session — downloading images will not clear results.")
