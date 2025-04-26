\documentclass[11pt]{article}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{AccidentAugment Roadmap (5-Week Plan)}
\date{}           % no date
\begin{document}
\maketitle

\section*{Phase 0 — One-Time Setup (Day 0–1)}
\begin{enumerate}[leftmargin=*, label=\arabic*.]
  \item \textbf{Create the repo skeleton}
    \begin{verbatim}
mkdir accidentaugment && cd accidentaugment
git init
# copy folder structure; commit README.md and .gitignore
    \end{verbatim}

  \item \textbf{Set up the Python environment}
    \begin{verbatim}
conda create -n aa python=3.10
conda activate aa
pip install torch torchvision torchaudio --extra-index-url \
        https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate bitsandbytes xformers==0.0.25
pip install lightning hydra-core wandb opencv-python tqdm
pip install scikit-image scikit-learn matplotlib seaborn
pip install lora-diffusion peft                # LoRA support
    \end{verbatim}

  \item \textbf{Smoke-test Stable Diffusion}
    \begin{verbatim}
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
          "runwayml/stable-diffusion-v1-5",
          torch_dtype="float16").to("cuda")
pipe("A clear highway under blue sky").images[0]
    \end{verbatim}
\end{enumerate}

\section*{Phase 1 — Few-Shot Data Prep (Day 1–3)}
\begin{itemize}[leftmargin=*]
  \item Collect \textasciitilde5–10 licence-free accident photos $\rightarrow$ \verb|data/raw/accident/|.
  \item (Optional) Run a pre-trained segmentation net, hand-fix masks $\rightarrow$ \verb|data/raw/accident_masks/|.
  \item Resize to $512\times512$, save PNGs $\rightarrow$ \verb|data/processed/accident/|.
  \item Track data via DVC or git-lfs; keep raw files out of Git history.
\end{itemize}

\section*{Phase 2 — Concept Tuning (Day 3–10)}
\begin{enumerate}[leftmargin=*, label=\arabic*.]
  \item \textbf{Choose a tuning method}

    \begin{tabular}{@{}lll@{}}
    \textbf{Method} & \textbf{VRAM} & \textbf{Pros / Cons}\\\hline
    Textual Inversion & $\le4$ GB & ultralight, lower fidelity\\
    LoRA fine-tune & 6–8 GB & good fidelity, few params\\
    DreamBooth & 10–12 GB & highest fidelity, slower\\
    \end{tabular}

  \item Prepare $\sim$300 regularization images (generic highways).
  \item Train LoRA\,:\\[-4pt]
\begin{verbatim}
python -m accidentaugment.fine_tune.train_lora \
  --pretrained runwayml/stable-diffusion-v1-5 \
  --train_data_dir data/processed/accident \
  --reg_data_dir data/regularization \
  --placeholder_token "<accident>" \
  --output_dir models/checkpoints/lora_accident_v1
\end{verbatim}
  \item Qualitative prompt check:\\
    \verb|pipe("A <accident> on a rainy urban street at night")|
\end{enumerate}

\section*{Phase 3 — Controllable Generation (Day 10–24)}
\begin{enumerate}[leftmargin=*, label=\arabic*.]
  \item Download or fine-tune a segmentation ControlNet.
  \item Build \texttt{gen\_batch.py}: combine Stable Diffusion + LoRA + ControlNet.  
        For each template mask, randomly insert damaged-car labels, vary weather, generate $N\!\approx\!1000$ PNGs $+$ masks.
  \item De-duplicate via CLIP similarity; manually review $\sim$100 outputs.
\end{enumerate}

\section*{Phase 4 — Downstream Evaluation (Day 24–32)}
\begin{enumerate}[leftmargin=*, label=\arabic*.]
  \item Select task: damaged-car \textit{detection} (YOLOv8) or \textit{segmentation} (DeepLab V3+).
  \item Train baseline on real data; record mAP/mIoU on real accident val set.
  \item Augment with $k=500\text{–}1000$ synthetic images; retrain; measure $\Delta$.
  \item Ablate “LoRA only” vs “LoRA + ControlNet”; test robustness (fog, rain).
\end{enumerate}

\section*{Phase 5 — Wrap-up (Day 32–35)}
\begin{itemize}[leftmargin=*]
  \item Push code, configs; DVC‐push data/checkpoints.
  \item Write 6–8-page report (figures, tables, sample grid of 16 images).
  \item Prepare 10–12-slide deck for class presentation.
  \item Provide demo notebook \texttt{04\_demo.ipynb}: mask $\rightarrow$ synthetic accident in $<30$ s.
\end{itemize}

\section*{Time-Saving Tips}
\vspace{-4pt}
\begin{itemize}[leftmargin=*]
  \item Finish Phase 0 in one sitting to stay motivated.
  \item Commit daily—even half-working code prevents lost work.
  \item Enable \verb|xformers| and Euler-a scheduler for 2–3$\times$ faster sampling on GTX 1080 Ti.
  \item Use WandB or Lightning logs from Day 1 to catch regressions early.
\end{itemize}

\end{document}
