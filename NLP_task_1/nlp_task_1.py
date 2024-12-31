import gzip
from typing import Optional, List
from dataclasses import dataclass
from yargy import Parser, rule, and_, or_
from yargy.pipelines import morph_pipeline
from yargy.predicates import dictionary, is_capitalized, gram, normalized, type as yargy_type
from yargy.interpretation import fact

# Define categories based on the assignment requirements
CATEGORIES = ['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']

@dataclass
class Entry:
    name: Optional[str]
    birth_date: Optional[str]
    birth_place: Optional[str]

# Define structures for names, dates, and places
Name = fact('Name', ['first', 'last'])
Date = fact('Date', ['day', 'month', 'year'])
Place = fact('Place', ['location'])

# Rules for recognizing names, dates, and places of birth
NAME = rule(
    and_(is_capitalized(), or_(gram('Name'), gram('Surn'))).interpretation(Name.first),
    and_(is_capitalized(), or_(gram('Name'), gram('Surn'))).interpretation(Name.last)
).interpretation(Name)

MONTHS = morph_pipeline([
    'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
])
DATE = rule(
    yargy_type('INT').optional().interpretation(Date.day.custom(str)),
    MONTHS.interpretation(Date.month.normalized().custom(str.lower)),
    yargy_type('INT').optional().interpretation(Date.year.custom(str))
).interpretation(Date)

PLACE = rule(
    dictionary({'город', 'село', 'поселок', 'деревня'}),
    is_capitalized().interpretation(Place.location)
).interpretation(Place)

# Birth context as a rule with interpretation
BIRTH_CONTEXT_RULE = rule(
    or_(
        normalized('родился'),
        normalized('рожден'),
        normalized('родилась'),
        normalized('рождена')
    )
)

parser_name = Parser(NAME)
parser_date = Parser(DATE)
parser_place = Parser(PLACE)
parser_birth_context = Parser(BIRTH_CONTEXT_RULE)

def extract_entry(content):
    entry = Entry(None, None, None)

    # Extract information only if birth context is present in the content
    birth_context_matches = list(parser_birth_context.findall(content.lower()))
    if birth_context_matches:
        name_matches = list(parser_name.findall(content))
        date_matches = list(parser_date.findall(content))
        place_matches = list(parser_place.findall(content))

        if name_matches:
            entry.name = ' '.join(name_matches[0].fact.as_json.values())

        if date_matches:
            date_parts = date_matches[0].fact.as_json
            # Only join the parts that exist, avoiding 'нет' for missing parts
            date_components = []
            if 'day' in date_parts:
                date_components.append(str(date_parts['day']))
            if 'month' in date_parts:
                date_components.append(str(date_parts['month']))
            if 'year' in date_parts:
                date_components.append(str(date_parts['year']))

            entry.birth_date = ' '.join(date_components) if date_components else 'нет'

        if place_matches:
            entry.birth_place = place_matches[0].fact.location

    return entry

def process_gzipped_text_file(file_path):
    entries_with_categories = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                category, title, content = parts
                if category in CATEGORIES:
                    entry = extract_entry(content)
                    if any([entry.name, entry.birth_date, entry.birth_place]):
                        entries_with_categories.append((category, entry))

    return entries_with_categories

def save_results(entries_with_categories, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (category, entry) in enumerate(entries_with_categories, 1):
            f.write(f"Параграф {i} извлеченные данные (Категория: {category}):\n")
            f.write(f"Имя: {entry.name if entry.name else 'нет'}\n")
            f.write(f"Дата рождения: {entry.birth_date if entry.birth_date else 'нет'}\n")
            f.write(f"Место рождения: {entry.birth_place if entry.birth_place else 'нет'}\n\n")

if __name__ == "__main__":
    news_file_path = 'D:/MyProject/SPBU Course/NLP/nlp_task_1/news.txt.gz'  # Update this path to where your .gz file is located
    output_file = 'extracted_entries.txt'

    print("Processing gzipped text file...")
    entries_with_categories = process_gzipped_text_file(news_file_path)
    print(f"Saving results to {output_file}")
    save_results(entries_with_categories, output_file)
    print("Processing completed.")