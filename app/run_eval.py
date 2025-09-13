import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import AzureChatOpenAI
from langchain.evaluation.qa import ContextQAEvalChain

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    temperature=0,
)
langchain_eval = ContextQAEvalChain.from_llm(llm)

# ✅ Establecer experimento una vez
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]
        contexto = result.get("context", "") 

        # Evaluación con LangChain
        graded = langchain_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada,
            context=contexto
        )

        # 🔍 Imprimir el contenido real
        print(f"\n📦 Resultado evaluación LangChain para pregunta {i+1}/{len(dataset)}:")
        print(graded)

        lc_verdict = graded.get("value", "UNKNOWN")
        is_correct = graded.get("score", 0)

        # Log en MLflow
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        mlflow.log_metric("lc_is_correct", is_correct)

        print(f"✅ Pregunta: {pregunta}")
        print(f"🧠 LangChain Eval: {lc_verdict}")