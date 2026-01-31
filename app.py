import streamlit as st
import os
import docx  # مكتبة التعامل مع ملفات الوورد
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # لتحويل نص الوورد لشكل يفهمه السيستم
from dotenv import load_dotenv
# 1. إعدادات الصفحة والـ CSS
st.set_page_config(page_title="AI Legal Auditor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #1e3d59;
        color: white;
        font-weight: bold;
        border: none;
        height: 3em;
    }
    .stButton>button:hover { background-color: #ffc107; color: #1e3d59; border: 1px solid #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# دالة مساعدة لقراءة ملفات الوورد
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    # تحويل النص إلى تنسيق Document الخاص بـ LangChain
    return [Document(page_content=text, metadata={"source": "uploaded_docx"})]

st.title("⚖️ منصة تدقيق العقود الذكية")
st.write("قم برفع العقد (PDF أو Word) للحصول على تحليل قانوني بميزان القانون المصري.")
load_dotenv()
# 2. تحميل النماذج
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="./my_model")
    llm = ChatGroq(
        temperature=0, 
        # التعديل هنا: اسم الباراميتر لازم يكون api_key
        api_key=os.getenv("GROQ_API_KEY"), 
        model_name="llama-3.3-70b-versatile"
    )
    return embeddings, llm

embeddings, llm = load_models()

# 3. رفع ومعالجة الملف (يدعم PDF و DOCX)
uploaded_file = st.file_uploader("ارفع العقد", type=["pdf", "docx"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner(f"جاري معالجة ملف {file_extension.upper()}..."):
        # حفظ ملف مؤقت للتعامل معه
        with open(f"temp.{file_extension}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # التبديل بين القارئ حسب نوع الملف
        if file_extension == "pdf":
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
        else:
            docs = read_docx(f"temp.docx")
        
        # تقسيم النص
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        # إنشاء الـ Vector Store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. دالة التحليل الذكية
    def run_legal_task(task_instruction, use_table=False):
        relevant_docs = retriever.invoke(task_instruction)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        table_instruction = "برجاء عرض النتائج في شكل جدول Markdown منظم (البند | التفاصيل)." if use_table else ""

        full_prompt = f"""أنت مستشار قانوني مصري خبير. استخدم المعلومات التالية فقط لتحليل العقد وفقاً لأصول القانون المصري.
        
        المعلومات المستخرجة:
        {context}
        
        المهمة: {task_instruction}
        {table_instruction}
        
        قواعد الإجابة:
        1. الإجابة بالعربية الفصحى فقط.
        2. ادخل في الموضوع مباشرة بدون مقدمات.
        3. كن دقيقاً جداً في استخراج الالتزامات والثغرات القانونية."""
        
        with st.spinner("يتم الآن الفحص القانوني..."):
            response = llm.invoke(full_prompt)
            return response.content

    # 5. عرض الخيارات
    st.write("---")
    st.subheader("إجراءات سريعة:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 ملخص العقد"):
            res = run_legal_task("لخص أهم 5 بنود في العقد (الأطراف، المدة، القيمة، الغرض، الفسخ).", use_table=True)
            st.markdown(res)

    with col2:
        if st.button("🚨 كشف المخاطر"):
            res = run_legal_task("استخرج الثغرات القانونية والمخاطر المحتملة حسب القانون المصري في شكل نقاط واضحة.")
            st.warning(res)

    with col3:
        if st.button("💰 الالتزامات"):
            res = run_legal_task("ما هي الالتزامات المالية والجدول الزمني المذكور؟", use_table=True)
            st.success(res)

    # 6. قسم الدردشة
    st.divider()
    st.subheader("💬 اسأل المستشار القانوني")
    user_query = st.text_input("مثلاً: ما هي شروط فسخ هذا العقد؟")
    if user_query:
        answer = run_legal_task(user_query)
        st.write("**الرد القانوني:**")
        st.write(answer)

else:
    st.info("💡 **جاهز للبدء:** ارفع عقدك الآن بصيغة **PDF** أو **Word**، وسيقوم المستشار الذكي باستخراج الثغرات والالتزامات في ثوانٍ.")