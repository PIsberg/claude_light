import sys
from unittest.mock import patch

class MockManager:
    def __init__(self, preset):
        from claude_light.ui import _T_TEST
        self._T_TEST = _T_TEST
        self.preset = preset
        self.files = {}
        self.total_tokens = 0
        self._generate_synthetic_files()
        
    def _generate_synthetic_files(self):
        configs = {"small": (5, 10), "medium": (50, 15), "large": (200, 20), "extra-large": (1000, 20)}
        num_files, num_methods = configs.get(self.preset, (5, 10))
        for i in range(num_files):
            file_name = f"src/main/java/com/synthetic/Service{i}.java"
            methods = []
            for m in range(num_methods):
                methods.append(f"""
    public void doTask{m}() {{
        System.out.println("Task {m} in Service {i}");
        for(int j=0; j<10; j++) {{
            // realistic logic simulated here
        }}
    }}""")
            content = f"""package com.synthetic;\n\nimport java.util.*;\n\npublic class Service{i} {{\n    private String name = "Service{i}";\n    {"".join(methods)}\n}}\n"""
            self.files[file_name] = content
            self.total_tokens += len(content) // 4
            
    def start(self):
        import claude_light.skeleton
        import claude_light.indexer
        import claude_light.editor
        import claude_light.llm
        import claude_light.executor
        import claude_light.config
        
        mock_path = self._mock_path_class()
        self.path_patchers = [
            patch.object(claude_light.skeleton, 'Path', new=mock_path),
            patch.object(claude_light.indexer, 'Path', new=mock_path),
            patch.object(claude_light.editor, 'Path', new=mock_path),
            patch.object(claude_light.config, 'Path', new=mock_path),
        ]
        self.path_patcher = self.path_patchers[0]
        for p in self.path_patchers: p.start()
        
        self.api_patcher = patch.object(claude_light.llm.client.messages, "create", side_effect=self._mock_create_message)
        self.api_patcher.start()
        
        self.orig_print_stats = claude_light.llm.print_stats
        self.stats_patcher = patch.object(claude_light.llm, "print_stats", side_effect=self._mock_print_stats)
        self.stats_patcher.start()
        
        self.embedder_patcher = patch.object(claude_light, "SentenceTransformer", new=self._mock_embedder_class)
        self.embedder_patcher.start()
        
        print(f"{self._T_TEST} Initialized '{self.preset}' preset with {len(self.files)} files (~{self.total_tokens:,} tokens).")
        
    def _mock_path_class(self):
        files = self.files
        class MockPath:
            def __init__(self, *args):
                self.path = "/".join(str(p) for p in args).replace("\\", "/")
                self.name = self.path.split("/")[-1]
                self.suffix = "." + self.name.split(".")[-1] if "." in self.name else ""
                self.stem = self.name[:-len(self.suffix)] if self.suffix else self.name
                self.parts = tuple(self.path.split("/"))
            
            def __str__(self): return self.path
            def __lt__(self, other): return self.path < getattr(other, "path", str(other))
            def __eq__(self, other): return self.path == getattr(other, "path", str(other))
            def __hash__(self): return hash(self.path)
            
            def rglob(self, pattern):
                for f in files: yield MockPath(f)
                    
            def read_text(self, *args, **kwargs):
                if self.path in files: return files[self.path]
                if self.name.endswith(".md"): return ""
                raise OSError(f"File not found: {self.path}")
                
            def read_bytes(self): return self.read_text().encode("utf-8")
            def is_file(self): return self.path in files or self.name.endswith(".md")
            def is_dir(self): return not self.is_file()
            def exists(self): return self.path in files or self.path in (".", ".claude_light_cache") or self.path.startswith("src")
            def stat(self):
                class Stat: st_size = len(files.get(self.path, ""))
                return Stat()
            def relative_to(self, other): return MockPath(self.path)
            def mkdir(self, *args, **kwargs): pass
            def write_text(self, *args, **kwargs): pass
            def write_bytes(self, *args, **kwargs): pass
            
        return MockPath

    def _mock_create_message(self, **kwargs):
        system_blocks = kwargs.get("system", [])
        messages = kwargs.get("messages", [])
        retrieved_ctx = ""
        
        for b in system_blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text = b.get("text", "")
                if "// src/" in text or "package com." in text:
                    retrieved_ctx += text
        
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                if "// src/" in content or "package com." in content:
                    retrieved_ctx += content
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text = b.get("text", "")
                        if "// src/" in text or "package com." in text:
                            retrieved_ctx += text
                    
        injected_tokens = len(retrieved_ctx) // 4
        full_tokens = self.total_tokens
        
        methods_mentioned = []
        if retrieved_ctx:
            import re
            matches = re.findall(r"public void (doTask\d+)", retrieved_ctx)
            if matches:
                methods_mentioned = list(set(matches[:3]))
                
        response_text = f"Simulated test response. Mentioning retrieved methods: {', '.join(methods_mentioned) if methods_mentioned else 'none'}."
        
        class MockUsage:
            def __init__(self, full, injected):
                self.input_tokens = injected
                self.cache_read_input_tokens = injected
                self.cache_creation_input_tokens = 0
                self.output_tokens = 50
                self._full_codebase_tokens = full
                self._injected_tokens = injected
                
        class MockMessage:
            def __init__(self, text, usage):
                class Content:
                    def __init__(self, t): self.text = t
                self.content = [Content(text)]
                self.usage = usage
                
        return MockMessage(response_text, MockUsage(full_tokens, injected_tokens))
        
    def _mock_embedder_class(self, model_name, **kwargs):
        class MockEmbedder:
            def encode(self, sentences, **kwargs):
                import numpy as np
                dim = 768 if "nomic" in model_name else 384
                if isinstance(sentences, str):
                    return np.random.rand(dim).astype(np.float32)
                return np.random.rand(len(sentences), dim).astype(np.float32)
        return MockEmbedder()
        
    def _mock_print_stats(self, usage, label="Stats", file=sys.stdout):
        self.orig_print_stats(usage, label, file)
        
        full_tokens = getattr(usage, "_full_codebase_tokens", self.total_tokens)
        injected = getattr(usage, "_injected_tokens", getattr(usage, "input_tokens", 0) + getattr(usage, "cache_read_input_tokens", 0))
        
        PRICE_INPUT  = 3.00
        PRICE_READ   = 0.30
        
        full_cost = (full_tokens / 1_000_000) * PRICE_INPUT
        rag_cost = (injected / 1_000_000) * PRICE_READ
        savings = full_cost - rag_cost
        savings_pct = (savings / full_cost * 100) if full_cost > 0 else 0.0
        
        print(f"\n[{label}] Token Savings Report:\n  If full codebase was sent: {full_tokens:,} tokens (${full_cost:.4f})\n  With Claude Light RAG + Cache: {injected:,} tokens (${rag_cost:.4f})\n  Total Savings: {savings_pct:.1f}%\n", file=file)
