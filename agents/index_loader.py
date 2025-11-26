from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

try:
    import faiss  # optional
    _FAISS_OK = True
except Exception:
    faiss = None
    _FAISS_OK = False

class RetrieverIndex:
    def __init__(self, store_dir: Path, embed_model_name: str):
        self.store_dir = Path(store_dir)
        self.embedder = SentenceTransformer(embed_model_name, device="cpu")

        self.index_faiss = self.store_dir / "index.faiss"
        self.meta_path   = self.store_dir / "metadata.parquet"
        self.docs_path   = self.store_dir / "documents.parquet"
        self.emb_npz     = self.store_dir / "embeddings.npz"  # used in NumPy fallback

        if not self.meta_path.exists() or not self.docs_path.exists():
            raise FileNotFoundError(f"Missing {self.meta_path.name} / {self.docs_path.name} in {self.store_dir}")

        self.meta = pd.read_parquet(self.meta_path)
        self.docs = pd.read_parquet(self.docs_path)

        self.backend = "faiss" if _FAISS_OK and self.index_faiss.exists() else "numpy"

        if self.backend == "faiss":
            self.index = faiss.read_index(str(self.index_faiss))
        else:
            if self.emb_npz.exists():
                self.emb = np.load(self.emb_npz)["emb"].astype(np.float32)
            else:
                texts = self.docs.sort_values("id")["text"].tolist()
                embs: List[np.ndarray] = []
                B = 512
                for i in range(0, len(texts), B):
                    batch = self.embedder.encode(
                        texts[i:i+B], batch_size=64,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False
                    ).astype(np.float32)
                    embs.append(batch)
                self.emb = np.vstack(embs)
                np.savez_compressed(self.emb_npz, emb=self.emb)
            self.meta = self.meta.sort_values("id").reset_index(drop=True)
            self.docs = self.docs.sort_values("id").reset_index(drop=True)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        rows: List[Dict[str, Any]] = []
        if self.backend == "faiss":
            sims, ids = self.index.search(q, k)
            sims, ids = sims[0], ids[0]
            id_to_row = self.meta.set_index("id")
            txt_map = self.docs.set_index("id")["text"]
            for s, i in zip(sims, ids):
                md = id_to_row.loc[int(i)].to_dict()
                rows.append({
                    "id": int(i),
                    "score": float(s),
                    "title": md.get("title",""),
                    "url": md.get("url",""),
                    "published": md.get("published",""),
                    "domain": md.get("domain",""),
                    "source": md.get("source",""),
                    "chunk": int(md.get("chunk",0)),
                    "text": txt_map.loc[int(i)]
                })
            return rows
        else:
            sims = (self.emb @ q[0])
            idx = np.argpartition(-sims, k)[:k]
            top = idx[np.argsort(-sims[idx])]
            for i in top:
                md = self.meta.iloc[int(i)].to_dict()
                rows.append({
                    "id": int(md["id"]) if "id" in md else int(i),
                    "score": float(sims[i]),
                    "title": md.get("title",""),
                    "url": md.get("url",""),
                    "published": md.get("published",""),
                    "domain": md.get("domain",""),
                    "source": md.get("source",""),
                    "chunk": int(md.get("chunk",0)),
                    "text": self.docs.iloc[int(i)]["text"]
                })
            return rows

    def info(self) -> str:
        return f"RetrieverIndex(backend={self.backend}, vectors={len(self.meta)})"
