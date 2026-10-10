import pymupdf
import pymssql
import json
import re
import os
from dotenv import load_dotenv
import ollama

load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
SERVER_HOST = os.getenv("SERVER_HOST")
SERVER_USER = os.getenv("SERVER_USER")
SERVER_PASS = os.getenv("SERVER_PASS")
SERVER_DB   = os.getenv("SERVER_DB")
WORD_LIMIT  = 500
DEFAULT_INSTRUCTION = """
You are a knowledge graph extraction engine for plantation company. Extract
subject-edge-object triples from the text for graph database insertion.

OUTPUT FORMAT
Return ONLY a valid JSON object: {{"triples": [...]}}
Each triple has exactly: "subject", "subject_type", "edge", "object", "object_type"
No relevant triples → return: {{"triples": []}}

PREDEFINED NODE TYPES (fixed spelling and capitalisation):
 Employee    — full name, Title Case (e.g. "John Smith")
 Vendor    — vendor name, Title Case (e.g. "Acme Supplies Ltd.")
 PurchaseOrder — PO ID (e.g. "PO-2024-001") or short Title Case phrase
 Approval    — approval ID (e.g. "APR-2024-005") or short Title Case phrase
 Department      — department full name (e.g. "Research Centre")

PREDEFINED EDGES:
 SUBMITTED, CREATED, APPROVED, REJECTED, REVIEWED,
 ISSUED_TO, RECEIVED_BY, FULFILLED_BY, REQUIRES_APPROVAL,
 HAS_APPROVAL, GRANTED_FOR, REQUESTED_BY, AUTHORIZED_BY,
 MANAGES, REPORTS_TO, DELEGATED_TO

EDGE CONSOLIDATION — map these to predefined edges, never create new ones for them:
 oversees / supervises / is responsible for → MANAGES
 signs / endorses / certifies / validates    → APPROVED
 evaluates / checks / audits / inspects    → REVIEWED
 sends / forwards / raises / files        → SUBMITTED
 assigns / hands over / transfers        → DELEGATED_TO
 needs / must have / depends on        → REQUIRES_APPROVAL
 purchases from / orders from / buys from    → ISSUED_TO
 is head of / leads / heads         → MANAGES
 works under / is subordinate to           → REPORTS_TO

CREATE TYPES — only when nothing above fits:
 BANNED — DO NOT CREATE THIS:
 - Generic placeholders: NewType, Node, NewNode, Entity, Object, Type, Item, Other
 - SQL keywords or common words: For, Procedure, By, On, In, To, From, Key, Table, Limit
 - Vague edges: RELATED_TO, ASSOCIATED_WITH, LINKED_TO, HAS, IS, FOR
 - Self-test: "Is this name domain-specific and unambiguous?" — if no, omit the triple.

 Node: PascalCase, domain-specific noun (e.g. Pesticide, Equipment, Regulation)
 Edge: UPPER_SNAKE_CASE, verb phrase (e.g. KILLED_BY, APPLIED_TO, CONTAINS)

 Before creating a new edge, ask:
 "Can any predefined or discovered edge express this, even loosely?" → if yes, use it.
 "Is this edge name a domain-specific verb unique to plantation operations?" → if no, omit.

 Before creating a new node, ask:
 "Can any predefined or discovered node express this, even loosely?" → if yes, use it.
 "Is this node name a domain-specific verb unique to plantation operations?" → if no, omit.

VALUE NAMING (subject and object fields):
 - Title Case always: "Ulat Bulu" not "ulatBulu" or "ULAT BULU"
 - No abbreviations unless they are official IDs (e.g. PO-2024-001)
 - Use English unless the term is a domain-specific proper noun
 - No double spaces, no trailing spaces

RULES
1. Prefer predefined or discovered types. Create new ones only as a last resort.
2. Only extract information explicitly stated in the text. No hallucination.
3. One triple per relationship. No duplicates.
4. Ignore dates, amounts, locations, and item descriptions.
5. Ignore header-only tables (e.g. those containing "No. Prosedur/Pedoman").
6. If an edge cannot be cleanly named, omit the triple entirely.
"""

client = ollama.Client(OLLAMA_HOST)
conn = pymssql.connect(SERVER_HOST, SERVER_USER, SERVER_PASS, SERVER_DB)
conn.autocommit(True)
cursor = conn.cursor()

def scrape_pdf(pdf_path: str) -> str:
    print(f"scraping: {pdf_path}")
    file = pymupdf.open(pdf_path)
    md_output = []

    for page_num in range(len(file.page_count)):
        page = file.load_page(page_num)
        print("Scraping page number: " + str(page_num))

        md_output.append(page.get_text("text"))
        try:
            tabs = page.find_tables()
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                md_output.append(df.to_markdown(index=False))
        except Exception as e:
            pass
    
    file.close()
    return "\n".join(md_output)

def convert_to_chunk(pdf_markdownf: str) -> list[str]:
    lines = pdf_markdownf.split('\n')
    final_chunks = []
    current_chunk = []
    current_word_count = 0

    current_is_table = False

    for line in lines:
        is_table_line = bool(re.match(r'^\s*\|', line))
        is_header = bool(re.match(r'^#+\s', line))
        line_word_count = len(line.split())

        if (current_is_table or (current_word_count + line_word_count < WORD_LIMIT) and (not is_header)):
            current_chunk.append(line)
            current_word_count += line_word_count
        else:
            if current_chunk:
                final_chunks.append("\n".join(current_chunk).strip())
            current_chunk = [line]
            current_word_count = line_word_count
        
        current_is_table = is_table_line

    if current_chunk:
        final_chunks.append("\n".join(current_chunk).strip())

    return final_chunks

def extract_triples_chunk(chunk: str) -> list[dict]:
    instruction = DEFAULT_INSTRUCTION
    prompt = f"""
Analyze the objects in this chunk and understand the relationship between objects.
Categorize the objects into the matching node name based on the instruction given.
DO NOT categorize too generally (e.g. Node and NewType are not valid categories, while Palm and Chemical are valid)
Extract all subject-edge-object triples from the following text.
Follow all instructions exactly. Return only the JSON object.

Text: 
{chunk}
    """

    response = client.chat(
        model = "qwen2.5:3b",
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        response_format = {"type": "json_object"},
        temperature=0
    )
    output = json.loads(response['message']['content'])
    print("New triple set created: ")
    print(output)
    return output.get("triples", [])

def process_pdf(pdf_path: str) -> None:
    md_content = scrape_pdf(pdf_path)
    print(f"Markdown format created for: {pdf_path}")

    chunks = convert_to_chunk(md_content)
    print(f"Markdown turned to {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
      new_triples = extract_triples_chunk(chunk)
      print(f"Triples for chunk {idx + 1} created")
      for triple in new_triples:
          upload_triple(triple)

def create_node(node_type: str, node_name: str) -> str:
    query = """
    IF NOT EXISTS (SELECT 1 FROM Nodes WHERE name = %s)
    BEGIN
        INSERT INTO Nodes (name, type) VALUES (%s, %s);
    END

    SELECT $node_id FROM Nodes WHERE name = %s;
    """
    cursor.execute(query, (node_name, node_name, node_type, node_name))
    
    row = cursor.fetchone()
    node_id = row[0] if row else None
    return node_id

def create_edge(subject_id: str, object_id: str, edge_name: str) -> None:
    query = """
    IF NOT EXISTS (
        SELECT 1 FROM Edges 
        WHERE $from_id = %s AND $to_id = %s AND name = %s
    )
    BEGIN
        INSERT INTO Edges ($from_id, $to_id, name)
        VALUES (%s, %s, %s);
    END
    """
    cursor.execute(query, (subject_id, object_id, edge_name, subject_id, object_id, edge_name))

def upload_triple(triple: dict) -> None:
    print("Uploading the following triple")
    print(triple)

    subject_id = create_node(triple['subject_type'], triple['subject'])
    object_id = create_node(triple['object_type'], triple['object'])
    create_edge(subject_id, object_id, triple['edge'])
    
def main():
    path = "MasterManual.pdf"
    process_pdf(path)

main()
