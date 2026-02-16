import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config
from ingest import KnowledgeBaseBuilder

class HuggingFaceAssistant:
    def __init__(self):
        if not os.path.exists(config.DB_PATH) or not os.listdir(config.DB_PATH):
            print(f"⚠️ База не найдена. Создаем индекс...")
            builder = KnowledgeBaseBuilder()
            builder.build_index()
        
        # 2. Инициализация
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_API_BASE,
            streaming=True
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=config.DB_PATH, 
            embedding_function=self.embeddings
        )
        
        self._setup_chains()

    def _setup_chains(self):
        # Router
        self.router_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "Classify the user question into: transformers, peft, accelerate, trl, datasets or general. Return ONLY the library name."),
                ("user", "{question}")
            ]) 
            | self.llm 
            | StrOutputParser()
        )

        # Fusion
        self.fusion_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "Generate 3 search queries for the user question. Return one per line."),
                ("user", "{question}")
            ]) 
            | self.llm 
            | StrOutputParser()
        )

        # Final RAG Prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are a Hugging Face expert. Answer based on the context.
        Context from {lib}:
        {context}
        
        Question: {question}
        Answer:
        """)

    def answer_stream(self, question: str):
        """
        Генератор, который возвращает поток событий:
        1. Логи процесса (Steps)
        2. Итоговый ответ по токенам
        """
        logs = ""
        
        # Step 1: routing
        yield "🔍 Analyzing request..."
        try:
            lib = self.router_chain.invoke(question).strip().lower()
            clean_lib = lib.replace("'", "").replace('"', "").replace(".", "")
            if clean_lib not in config.LIBS_CONFIG:
                clean_lib = "general"
        except:
            clean_lib = "general"
            
        logs += f"> 🧭 **Routing**: Selected library `{clean_lib}`\n"
        yield logs

        # Step2: Fusion
        yield logs + "> 🧠 **Planning**: Generating search strategies..."
        try:
            queries_raw = self.fusion_chain.invoke(question)
            queries = [q for q in queries_raw.split("\n") if q.strip()]
        except:
            queries = []
        queries.append(question)
        
        logs += f"> 🧠 **Strategy**: Generated {len(queries)} search queries\n"
        yield logs

        # Step 3: Retreival
        yield logs + f"> 📚 **Retrieval**: Searching in `{clean_lib}` docs..."
        
        search_kwargs = {"k": 4}
        if clean_lib != "general":
            search_kwargs["filter"] = {"lib_name": clean_lib}
            
        all_docs = []
        for q in queries:
            all_docs.extend(self.vectorstore.similarity_search(q, **search_kwargs))
        
        # Fallback
        if not all_docs and clean_lib != "general":
            logs += f"> ⚠️ **Retry**: No docs found in {clean_lib}, switching to global search...\n"
            yield logs
            clean_lib = "general"
            all_docs = self.vectorstore.similarity_search(question, k=5)

        unique_docs = {d.page_content: d for d in all_docs}.values()
        context_text = "\n\n".join([d.page_content for d in unique_docs])
        
        logs += f"> ✅ **Context**: Found {len(unique_docs)} relevant fragments\n\n---\n"
        yield logs

        # Step 4: Streaming (generation)
        chain = self.rag_prompt | self.llm | StrOutputParser()
        
        full_response = logs
        for chunk in chain.stream({
            "context": context_text,
            "question": question,
            "lib": clean_lib
        }):
            full_response += chunk
            yield full_response
            