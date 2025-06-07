from flask import Flask, render_template, request, Response
import openai
import json
import pandas as pd
import os
import re
import webbrowser
import threading
from functools import lru_cache
import io
import csv
from docx import Document
from dotenv import load_dotenv

# Percorso assoluto al file .env
env_path = os.path.join(os.path.dirname(__file__), 'templates', 'Key.env')
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Inizializza l'app Flask
app = Flask(__name__, template_folder="templates", static_folder="static")

def load_iqvia_data():
    base_path = "static/iqvia_data"
    markets = ['Germany', 'Italy', 'Spain', 'UK']
    all_data = []
    failed_files = []

    country_map = {
        'Italia': 'Italy', 'IT': 'Italy',
        'Germany': 'Germany', 'Deutschland': 'Germany',
        'España': 'Spain', 'ES': 'Spain',
        'UK': 'UK', 'United Kingdom': 'UK', 'Regno Unito': 'UK'
    }

    for market in markets:
        market_path = os.path.join(base_path, market)
        if not os.path.exists(market_path):
            continue
        for filename in os.listdir(market_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(market_path, filename)
                try:
                    df = pd.read_excel(file_path, engine="openpyxl", nrows=500)
                    if df.empty or df.shape[1] < 5:
                        raise ValueError("File vuoto o con troppe poche colonne")
                    df = df.iloc[:-1]
                    df.columns = df.columns.str.replace('\n', ' ', regex=True).str.replace(' +', ' ', regex=True).str.strip()
                    rename_map = {
                        'Principio attivo': 'Molecule',
                        'Prodotto': 'Product',
                        'Paese': 'Country',
                        '2022 EUR MNF': '2022 EUR MNF',
                        '2023 EUR MNF': '2023 EUR MNF',
                        '2024 EUR MNF': '2024 EUR MNF',
                        '2022 Standard Units': '2022 SU',
                        '2023 Standard Units': '2023 SU',
                        '2024 Standard Units': '2024 SU',
                        'Azienda': 'Corporation'
                    }
                   df.rename(columns=rename_map, inplace=True)
                    df['Country'] = df['Country'].astype(str).str.strip()
                    df['Market'] = market
                    all_data.append(df)
                except Exception as e:
                    print(f"❌ Errore nel file {file_path}: {e}")
                    failed_files.append(filename)
                    continue

    if not all_data:
        print("⚠️ Nessun file Excel valido caricato.")
        return pd.DataFrame()

    if failed_files:
        print(f"❌ I seguenti file non sono stati caricati: {', '.join(failed_files)}")

    print(f"✅ Totale file caricati con successo: {len(all_data)}")
    return pd.concat(all_data, ignore_index=True)

IQVIA_DATA = load_iqvia_data()

@app.route('/', methods=['GET'])
def homepage():
    filters = get_dynamic_filters(IQVIA_DATA)
    return render_template('index.html', risultati=[], risposta_generica=None, request=request, **filters)

def evaluate_competition(df, molecule):
    return (
        "High" if df[df['Molecule'] == molecule]['Corporation'].nunique() >= 6 else
        "Mid" if df[df['Molecule'] == molecule]['Corporation'].nunique() >= 4 else
        "Low"
    )

def extract_field(lines, label):
    for line in lines:
        if line.lower().startswith(label.lower()):
            return line.split(":", 1)[1].strip()
    return None

@lru_cache(maxsize=128)
def query_gpt_estimates(brand_name):
    prompt = (
        f"Provide the following information for the drug '{brand_name}':\n"
        f"Patent status in Europe: ...\n"
        f"Therapeutic area: ...\n"
        f"Originator company: ...\n"
        f"Respond concisely and avoid saying 'not available'. If uncertain, provide plausible estimates."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a pharmaceutical analyst. Provide factual or plausible estimates."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        reply = response['choices'][0]['message']['content']
        lines = reply.strip().split('\n')
        patent = extract_field(lines, "Patent status in Europe") or "Likely expired"
        area = extract_field(lines, "Therapeutic area") or "General medicine"
        originator = extract_field(lines, "Originator company") or "Estimated originator"
        return patent, area, originator
    except Exception as e:
        print(f"GPT error for {brand_name}: {e}")
        return "Estimate not available", "General medicine", "Generic company"
def is_structured_query(text):
    return bool(re.search(r"\b\d+\b.*\b(low|mid|high)\b.*\bcompetition\b.*\b(in|for)\b.*\b(italy|germany|spain|uk)\b", text.lower()))

def get_dynamic_filters(df):
    return {
        "molecule_names": sorted(df['Molecule'].dropna().unique()),
        "countries": sorted(df['Country'].dropna().unique()),
    }

def normalize_molecule_name(name):
    return re.sub(r'\W+', '', name.strip().lower()) if isinstance(name, str) else ""

def remove_duplicate_market_sizes(results):
    seen_inn = set()
    unique_results = []
    for item in results:
        if 'Active' in item['patent']:
            continue
        if item['inn'] not in seen_inn:
            seen_inn.add(item['inn'])
            unique_results.append(item)
    return unique_results

def local_opportunity_finder(iqvia_df, country=None, min_market_size=100000, limit=5,
                              competition_filter=None, formulation_filter=None,
                              therapeutic_area_filter=None, molecule_type_filter=None):
    df = iqvia_df.copy()
    df['Normalized_Molecule'] = df['Molecule'].apply(normalize_molecule_name)

    if country:
        df = df[df['Country'].str.lower() == country.lower()]
    if '2024 EUR MNF' not in df.columns:
        return []

    df = df[pd.notna(df['2024 EUR MNF']) & (df['2024 EUR MNF'] > 0)]

    if formulation_filter:
        df = df[df['Product'].str.contains(formulation_filter, case=False, na=False)]
    if therapeutic_area_filter:
        df = df[df['Product'].str.contains(therapeutic_area_filter, case=False, na=False)]
    if molecule_type_filter:
        df = df[df['Product'].str.contains(molecule_type_filter, case=False, na=False)]

    grouped = df.groupby(['Normalized_Molecule', 'Country']).agg({
        '2024 EUR MNF': 'sum',
        'Corporation': pd.Series.nunique,
        'Molecule': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        'Product': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
    }).reset_index()

    grouped = grouped[grouped['2024 EUR MNF'] >= min_market_size]
    grouped = grouped.sort_values(by='2024 EUR MNF', ascending=False)

    results = []
    used_molecules = set()
    idx = 0

    while len(results) < limit and idx < len(grouped):
        row = grouped.iloc[idx]
        idx += 1

        normalized = row['Normalized_Molecule']
        if normalized in used_molecules:
            continue

        molecule = row['Molecule']
        brand_name = row['Product']
        market_value = int(row['2024 EUR MNF'])
        country = row['Country']

        subset_df = df[(df['Normalized_Molecule'] == normalized) & (df['Country'] == country)]
        competition = evaluate_competition(subset_df, molecule)
        if competition_filter and competition != competition_filter:
            continue

        patent, area, originator = query_gpt_estimates(brand_name)
        if any(p in patent.lower() for p in ["active until", "valid until", "still under patent"]):
            continue

        results.append({
            "inn": molecule,
            "brand": brand_name,
            "companies": int(row['Corporation']),
            "patent": patent,
            "competition": competition,
            "market_size": f"{market_value:,}".replace(",", ".") + " EUR",
            "notes": f"Therapeutic area: {area}\nOriginator: {originator}"
        })

        used_molecules.add(normalized)

    return results

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form.get('search_query', '')
    therapeutic_area = request.form.get('therapeutic_area', '')
    formulation = request.form.get('formulation', '')
    target_country = request.form.get('target_country', '')
    molecule_type = request.form.get('molecule_type', '')
    market_size = request.form.get('market_size', '')
    patent_status = "expired"  # default

    try:
        market_size = int(market_size)
    except:
        market_size = 100000

    filters = get_dynamic_filters(IQVIA_DATA)

    if not is_structured_query(search_query):
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical market expert."},
                    {"role": "user", "content": search_query}
                ],
                max_tokens=500,
                temperature=0.6
            )
            risposta_generica = gpt_response['choices'][0]['message']['content']
        except Exception as e:
            risposta_generica = f"Errore durante la risposta: {e}"
        return render_template('index.html', risultati=[], risposta_generica=risposta_generica, request=request, **filters)

    # Parsing avanzato della query
    num_results = 5
    competition_filter = None

    try:
        extraction_prompt = (
            f"Extract the structured parameters from this user query:\n"
            f"Query: '{search_query}'\n\n"
            f"Return a JSON with keys: 'country', 'competition', 'number', 'patent_status'.\n"
            f"Example:\n"
            f"{{\"country\": \"Italy\", \"competition\": \"Low\", \"number\": 5, \"patent_status\": \"expired\"}}"
        )
        extraction_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract clean JSON data from user queries about pharmaceutical opportunities."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        extracted = json.loads(extraction_response['choices'][0]['message']['content'])
        print("📦 Parametri estratti da GPT:", extracted)

        target_country = extracted.get("country", target_country)
        competition_filter = extracted.get("competition", competition_filter)
        num_results = int(extracted.get("number", num_results))
        patent_status = extracted.get("patent_status", "expired").lower()
    except Exception as e:
        print(f"⚠️ Errore nel parsing della query: {e}")

    risultati = local_opportunity_finder(
        IQVIA_DATA,
        country=target_country,
        min_market_size=market_size,
        limit=num_results,
        competition_filter=competition_filter,
        formulation_filter=formulation,
        therapeutic_area_filter=therapeutic_area,
        molecule_type_filter=molecule_type
    )

    # Filtro extra su brevetto
    risultati_filtrati = []
    for r in risultati:
        p = r.get("patent", "").lower()
        if patent_status == "expired" and "active" in p and "expired" not in p:
            continue
        if patent_status == "active" and "expired" in p:
            continue
        risultati_filtrati.append(r)

    return render_template('index.html', risultati=risultati_filtrati, risposta_generica=None, request=request, **filters)



@app.route('/export', methods=['POST'])
def export():
    results_json = request.form.get('results_json')
    if not results_json:
        return "No data to export", 400
    try:
        data = json.loads(results_json)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["inn", "patent", "competition", "companies", "market_size", "notes"], extrasaction='ignore')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        response = Response(output.getvalue(), mimetype='text/csv')
        response.headers.set("Content-Disposition", "attachment", filename="pharma_opportunities.csv")
        return response
    except Exception as e:
        return f"Errore durante l'esportazione: {e}", 500

@app.route('/export_excel', methods=['POST'])
def export_excel():
    results_json = request.form.get('results_json')
    if not results_json:
        return "No data to export", 400
    try:
        data = json.loads(results_json)
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
        output.seek(0)
        response = Response(output.read(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response.headers.set("Content-Disposition", "attachment", filename="pharma_opportunities.xlsx")
        return response
    except Exception as e:
        return f"Errore durante l'esportazione Excel: {e}", 500

@app.route('/export_word', methods=['POST'])
def export_word():
    word_text = request.form.get('word_text', '')
    if not word_text:
        return "No text to export", 400
    try:
        doc = Document()
        for paragraph in word_text.split('\n'):
            doc.add_paragraph(paragraph)
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)
        response = Response(output.read(), mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        response.headers.set("Content-Disposition", "attachment", filename="gpt_response.docx")
        return response
    except Exception as e:
        return f"Errore durante l'esportazione Word: {e}", 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, use_reloader=False)


