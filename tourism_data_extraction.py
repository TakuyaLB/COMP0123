import pdfplumber
import pandas as pd
import os

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def title_remove(country):
    titles = ['democratic', 'people', 'state', 'republic']
    country_parts = country.split(',')
    if len(country_parts) > 1:
        second_part = country_parts[1].lower()
        if any(map(second_part.__contains__, titles)):
            return country_parts[0]
        else:
            return False
    return country

# Path to the PDF file
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
pdf_path = os.path.join(__location__, "tourism_yearbook_ed2023_2.pdf")

# Initialize an empty list to store edges
edges = []

not_countries = ["american samoa", "anguilla", "aruba", "benelux", "bermuda", "bonaire", "british indian ocean territory", "british virgin islands", "cayman islands", "channel islands", "cocos (keeling) islands", "commonwealth independent states", "cook islands", "curaçao", "dubai", "faeroe islands", "falkland islands (malvinas)", "french guiana", "french polynesia", "gibraltar", "guadeloupe", "guam", "holy see", "isle of man", "johnston island", "martinique", "midway islands", "montserrat", "netherlands antilles", "new caledonia", "niue", "norfolk island", "northern mariana islands", "pitcairn", "puerto rico", "reunion", "saint helena", "saint pierre and miquelon", "sint maarten (dutch part)", "svalbard and jan mayen islands", "tokelau", "turks and caicos islands", "united states virgin islands", "wallis and futuna islands"]

# Open the PDF
with pdfplumber.open(pdf_path) as pdf:
    edges = {}
    for page in pdf.pages:
        # Extract source country for each page
        page_text = page.extract_text()
        parts = page_text.split()
        
        target_country = ''
        i = 3
        part = parts[i]
        while part != '/' and not is_float(part):
            target_country = target_country + ' ' + part 
            i += 1
            part = parts[i]
        target_country = target_country.strip()       
        target_country = title_remove(target_country)      
        
        # Process table for each page
        table = page.extract_table()
        if table and target_country.lower() not in not_countries:
            print(target_country)
        
            # Use the first row as header
            df = pd.DataFrame(table[1:], columns=table[0])  
            
            # Iterate over rows to handle missing data
            for _, row in df.iterrows():
                if not row[0].isupper() and row[0] != '': 
                    source_country = row.get("")
                    source_country = source_country.strip()
                    source_country = source_country.replace("\n", " ")
                    weight = row.get("2021")
                    # remove commas from large numbers for input parsing
                    weight = ''.join(c for c in weight if c != ',') 
                    
                    # Handle missing values
                    if weight is None or weight == "":
                        continue
                    else:
                        try:
                            weight = int(weight)
                        except ValueError:
                            continue
                    if source_country is None or source_country == "" or '(cid' in source_country or '/' in source_country:
                        continue
                    if "other countries" in source_country.lower() or "all countries" in source_country.lower() or "former" in source_country.lower():
                        continue
                    
                    if source_country.lower() == "hong kong, china":
                        source_country = "hong kong (china)"
                    if source_country.lower() == "macao, china":
                        source_country = "macao (china)"
                    if source_country.lower() == "czech republic":
                        source_country = "czech republic (czechia)"
                    
                    source_country = source_country.removesuffix(' (*)')
                    
                    remove_list = ["serbia and montenegro", "nationals residing abroad", "western sahara", "baltic countries", "scandinavia"]
                    if source_country.lower() in remove_list:
                        continue
                    
                    source_country = title_remove(source_country)
                    
                    if not source_country:
                        continue
                    
                    if source_country.lower() in not_countries:
                        continue
                    
                    # Use a tuple as the key for edges
                    edge_key = (source_country.lower(), target_country.lower()) # order swapped since data is arrivals
                    
                    # Add or update the edge with the highest weight
                    if edge_key not in edges or edges[edge_key] < weight:
                        edges[edge_key] = weight

# Convert to DataFrame
edge_list = pd.DataFrame([{"Source": key[0], "Target": key[1], "Weight": value} for key, value in edges.items()])

# Path to the CSV
csv_path = os.path.join(__location__, "tourism_network_2021.csv")

# Save as CSV for Gephi
edge_list.to_csv(csv_path, index=False)

print("Edge list saved as 'tourism_network_2021.csv'")


