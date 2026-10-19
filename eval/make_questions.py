"""Writes eval/questions.jsonl: 40 answerable questions and 10 the publications can't answer.

Every expected answer was read off the cleaned page text. Instead of typing page numbers by
hand, each question lists `evidence`: short exact sentences from the publications. This
script finds every page that contains them, so the expected pages are complete (the mileage
rate is stated on three pages) and can be re-derived if the cleaning changes.

`evidence` is a list of groups. A question needing two facts has two groups; a page answers
a group if it contains any sentence in it. Matching ignores spaces, hyphens, quote style and
case, the same way citation checks do, so PDF splits like "operat- ing" don't matter.

10 questions are held out (split "test"): don't look at their results until the final run.

Run from the project folder:  py -m eval.make_questions
"""
import json
import re
from pathlib import Path

from rag.chunk import load_chunks
from rag.pipeline import normalize

# (id, split, question, expected answer, evidence groups)
ANSWERABLE = [
    # ---- Pub 463: travel, gift and car expenses ----
    ("q01", "dev", "What is the standard mileage rate for business use of a car in 2025?",
     "70 cents ($0.70) per mile.",
     [["70 cents ($0.70) per mile", "standard mileage rate for 2025 is 70 cents"]]),
    ("q02", "dev", "How much can I deduct for business gifts I give to a client?",
     "No more than $25 per person per year for gifts given directly or indirectly.",
     [["You can deduct no more than $25 for business gifts you give directly or indirectly to each person"]]),
    ("q03", "dev", "What percentage of business meal costs can I deduct?",
     "Generally 50% of the cost (the 50% limit), if the meal isn't lavish or extravagant.",
     [["you can generally deduct only 50% of the unreimbursed cost of your meals",
       "deduct 50% of the cost of business meals",
       "meal expenses generally remain limited to 50%"]]),
    ("q04", "test", "Can I use the standard mileage rate if my business uses five cars at the same time?",
     "No. You can't use the standard mileage rate if you use five or more cars at the same time, "
     "such as in fleet operations; you may be able to deduct actual expenses instead.",
     [["five or more cars that are used for business at the same time, you can't use the standard mileage rate",
       "five or more cars at the same time"]]),
    ("q05", "dev", "If I use the standard mileage rate in 2025, what is the rate, and can I also deduct "
                   "parking fees and tolls?",
     "70 cents per mile; and yes, business-related parking fees and tolls are deductible in addition "
     "(but not parking at your regular place of work, which is commuting).",
     [["70 cents ($0.70) per mile", "standard mileage rate for 2025 is 70 cents"],
      ["In addition to using the standard mileage rate, you can deduct any business-related parking fees and tolls"]]),
    ("q06", "dev", "Where is my tax home for travel expense purposes?",
     "Generally your regular place of business or post of duty, regardless of where you keep your "
     "family home; it includes the entire city or general area.",
     [["your tax home is your regular place of business or post of duty, regardless of where you maintain your family home",
       "your tax home is your regular place of business, regardless of where you maintain your family home"]]),
    ("q07", "dev", "Do I need a receipt for a $50 business expense that isn't lodging?",
     "No documentary evidence (such as a receipt) is needed if the expense, other than lodging, is less "
     "than $75, though you still need the other records.",
     [["Your expense, other than lodging, is less than $75"]]),
    ("q08", "test", "Does a $3 pen with my company name printed on it count toward the business gift limit?",
     "No. An item costing $4 or less with your name clearly and permanently imprinted, one of many "
     "identical items you widely distribute, isn't considered a gift for the $25 limit.",
     [["An item that costs $4 or less"]]),
    ("q09", "dev", "How much business use do I need to claim a section 179 deduction for my car?",
     "More than 50% business use.",
     [["You must use the property more than 50% for business to claim any section 179 deduction"]]),

    # ---- Pub 587: business use of your home ----
    ("q10", "dev", "How does the simplified method for the home office deduction work?",
     "Multiply $5 by the square footage used for business, limited to 300 square feet "
     "(so at most $1,500).",
     [["figure your deduction by multiplying $5", "deduction by multiplying $5 by the area"],
      ["The area you use to figure your deduction is limited to 300 square feet"]]),
    ("q11", "dev", "Can I deduct a home office if I also use the room for personal purposes?",
     "Generally no: you don't meet the exclusive use test if you use the area both for business "
     "and personal purposes (exceptions: inventory storage and daycare).",
     [["exclusive use test if you use the area in question both for business and for personal purposes"]]),
    ("q12", "dev", "I store inventory in part of my basement and sometimes use that space personally. "
                   "Can I still deduct it?",
     "Yes. Storage of inventory or product samples is an exception to the exclusive use test, if your "
     "home is the only fixed location of a retail or wholesale business and you use the space regularly.",
     [["If you use part of your home for storage of inventory or product samples, you can deduct expenses "
       "for the business use of your home without meeting the exclusive use test"]]),
    ("q13", "test", "What does the regular use test for a home office require?",
     "You must use a specific area of your home for business on a regular (continuing) basis; "
     "incidental or occasional business use isn't regular use.",
     [["Incidental or occasional business use is not regular use",
       "you must use a specific area of your home for business on a continuing basis"]]),
    ("q14", "dev", "I run a daycare in a room my family also uses in the evenings. How do I figure the "
                   "deduction?",
     "Because daycare use is regular but not exclusive, figure the percentage of time that part of the "
     "home is used for daycare (hours used for daycare divided by total hours) and apply it.",
     [["you must figure the percentage of time that part of your home is used for daycare"]]),
    ("q15", "dev", "Over how many years do I depreciate the business part of my home if I started using "
                   "it for business in 2025?",
     "39 years, straight line, as nonresidential real property under MACRS.",
     [["nonresidential real property is depreciated using the straight line method over 39 years",
       "depreciate that part of your home as nonresidential real property over 39 years"]]),
    ("q16", "dev", "What MACRS property class are a computer and a desk used in a home office?",
     "A computer is 5-year property; office furniture such as a desk is 7-year property.",
     [["5-year property includes computers and peripheral equipment"],
      ["7-year property includes office furniture and fixtures such as desks",
       "Office furniture and fixtures (such as desks, files, and safes)"]]),

    # ---- Pub 946: depreciation ----
    ("q17", "dev", "How long is the recovery period for residential rental property?",
     "27.5 years.",
     [["Residential rental property ...... 27.5 years", "residential rental property over 27.5 years",
       "27.5-year residential rental property"]]),
    ("q18", "dev", "Can I depreciate the cost of land?",
     "No. Land doesn't wear out, become obsolete, or get used up, though some land preparation "
     "costs such as landscaping may be depreciable.",
     [["You cannot depreciate the cost of land because land does not wear out"]]),
    ("q19", "test", "What is the maximum section 179 deduction for 2025?",
     "$2,500,000, reduced by the amount by which section 179 property placed in service exceeds $4,000,000.",
     [["The maximum amount you can elect to deduct during 2025 is generally $2,500,000",
       "can't be more than $2,500,000", "cannot exceed $2,500,000"]]),
    ("q20", "dev", "What is the bonus (special) depreciation rate for qualified property placed in "
                   "service after January 19, 2025?",
     "100% (unless you elect out; a 40% election is available).",
     [["you must take a 100% special depreciation allowance",
       "is 100% for qualified property acquired and placed in service after January 19, 2025",
       "reinstated the 100% special depreciation allowance"]]),
    ("q21", "dev", "What does the half-year convention mean for depreciation?",
     "All property placed in service or disposed of during the year is treated as placed in service or "
     "disposed of at the midpoint of the year, so you get a half-year of depreciation in the first year.",
     [["which allows you to deduct a half-year of depreciation in the first year",
       "you treat all property placed in service or disposed of during a tax year as placed in service or disposed of at the midpoint"]]),
    ("q22", "dev", "When do I have to use the mid-quarter convention instead of the half-year convention?",
     "When more than 40% of the depreciable property placed in service during the year was placed in "
     "service in the last 3 months of the tax year.",
     [["more than 40% of your depreciable property in service during the last 3 months of your tax year",
       "during the last 3 months of the tax year is more than 40%"]]),
    ("q23", "dev", "At what total cost of section 179 property can I no longer take any section 179 "
                   "deduction in 2025?",
     "$6,500,000 or more.",
     [["is $6,500,000 or more, you can't take a section 179 deduction", "$6,500,000"]]),
    ("q24", "test", "Under the half-year convention, what percentage of a 5-year property's basis is "
                    "depreciated in the first year?",
     "20% (20.00%); for 7-year property it's 14.29%.",
     # p587 p17 has a clean table row. Pub 946's Table A-1 (p. 71) is extracted as number soup
     # ("... 7.41 20.00% 32.00 19.20 ..."), so it's listed directly instead of matched.
     [["1 20.00% 14.29%"]]),

    # ---- Pub 334: small business ----
    ("q25", "dev", "What is the self-employment tax rate for 2025?",
     "15.3%: 12.4% social security plus 2.9% Medicare.",
     [["15.3% (12.4% social security tax plus 2.9% Medicare tax)"]]),
    ("q26", "dev", "What is the maximum amount of net earnings subject to the social security part of "
                   "self-employment tax in 2025?",
     "$176,100 (there's no limit for the Medicare part).",
     [["Only the first $176,100 of your combined wages, tips, and net earnings in 2025",
       "subject to the social security part of the self-employment tax is $176,100"]]),
    ("q27", "dev", "I made $500 from self-employment this year. Do I have to file a tax return?",
     "Yes: you must file if your net earnings from self-employment were $400 or more.",
     [["if your net earnings from self-employment were $400 or more",
       "net earnings from self-employment of $400 or more", "if your net earnings from self-employment are $400 or more"]]),
    ("q28", "dev", "Can I deduct part of my self-employment tax?",
     "Yes, one-half of it, as an adjustment to income on Schedule 1 (Form 1040), line 15.",
     [["one-half of your SE tax", "one-half of your self-employment tax"]]),
    ("q29", "test", "How do statutory employees report their wages and expenses?",
     "On Schedule C (Form 1040); a statutory employee has box 13 checked on Form W-2.",
     [["Statutory employees use Schedule C (Form 1040) to report their wages and expenses"]]),
    ("q30", "dev", "When do I report rent I pay for my business on Form 1099-MISC?",
     "For rent payments of $600 or more, other than rent paid to real estate agents.",
     [["Rent payments of $600 or more, other than rents paid to real estate agents"]]),
    ("q31", "dev", "Can I deduct what I pay to park at my regular workplace?",
     "No. Parking fees at your place of work are nondeductible commuting expenses.",
     [["Parking fees you pay to park your car at your place of work are nondeductible commuting expenses"]]),

    # ---- Pub 17: individual income tax ----
    ("q32", "dev", "What is the 2025 standard deduction for married couples filing jointly?",
     "$31,500.",
     [["Married filing jointly or Qualifying surviving spouse-$31,500", "Their standard deduction is $31,500"]]),
    ("q33", "test", "What is the 2025 standard deduction for a single filer?",
     "$15,750.",
     [["Single or Married filing separately-$15,750"]]),
    ("q34", "dev", "When is my 2025 tax return due, and how long can I get with an extension?",
     "April 15, 2026; an automatic 6-month extension moves the filing date to October 15, 2026 "
     "(but tax owed is still due April 15).",
     [["File Form 1040 or 1040-SR by April 15, 2026", "April 15, 2026, is the due date for filing your 2025 income tax return"],
      ["you will have until October 15, 2026, to file"]]),
    ("q35", "dev", "How much can I contribute to a traditional IRA for 2025?",
     "$7,000, or $8,000 if you're 50 or older (limited to your taxable compensation).",
     [["$7,000 ($8,000 if you are 50 or older)"]]),
    ("q36", "dev", "What is the maximum child tax credit per qualifying child for 2025?",
     "$2,200 per qualifying child.",
     [["The maximum amount of CTC for each qualifying child increased to $2,200",
       "may reduce your tax by as much as $2,200 for each child"]]),
    ("q37", "dev", "How old can a child be and still be my qualifying child?",
     "Under 19 at the end of the year (and younger than you), or under 24 if a full-time student, "
     "or any age if permanently and totally disabled.",
     [["under age 19 at the end of the year and younger than you"]]),
    ("q38", "test", "Is jury duty pay taxable?",
     "Yes, it must be included in income (Schedule 1, line 8h); if you gave it to your employer, "
     "you can deduct that amount as an adjustment.",
     [["Jury duty pay you receive must be included in your income"]]),
    ("q39", "dev", "Are gambling winnings taxable, and can I deduct my gambling losses?",
     "Winnings must be included in income (Schedule 1, line 8b); losses are deductible only as an "
     "itemized deduction on Schedule A, up to the amount of winnings.",
     [["You must include your gambling winnings in income", "You must report the full amount of your gambling winnings"],
      # Not the section title: p. 78 only cross-references it ("see Gambling Losses up to...").
      ["You deduct your gambling losses for the year on Schedule A"]]),
    ("q40", "dev", "How long do I have to file a claim for a refund?",
     "Generally within 3 years from the date you filed the original return or 2 years from the date "
     "you paid the tax, whichever is later.",
     [["within 3 years from the date you filed your original return or 2 years from the date you paid the tax"]]),
]

# Answer pages that text matching can't find (tables flattened by PDF extraction), per group.
EXTRA_PAGES = {
    "q24": [["p946-p71"]],      # Table A-1, 5-year column: 20.00% in year 1
}

UNANSWERABLE = [
    ("u01", "dev", "What is the tax rate on lottery winnings in Canada?"),
    ("u02", "dev", "What is California's state income tax rate on capital gains?"),
    ("u03", "dev", "What will the standard mileage rate be for 2027?"),
    ("u04", "dev", "What is the federal corporate income tax rate for C corporations?"),
    ("u05", "test", "What is the 2025 federal estate tax exemption amount?"),
    ("u06", "dev", "How much can I contribute to a health savings account for 2025?"),
    ("u07", "dev", "When is Form 1120-S due for an S corporation?"),
    ("u08", "dev", "What is the minimum wage in New York?"),
    ("u09", "test", "Do I need to collect sales tax on items I sell on Etsy to customers in Texas?"),
    ("u10", "dev", "How do I file a UK Self Assessment tax return?"),
]


def squash(text: str) -> str:
    return re.sub(r"[\s-]+", "", normalize(text)).lower()


def main():
    pages = load_chunks("data/pages.jsonl")
    squashed = [(p["doc_id"], p["page"], squash(p["text"])) for p in pages]
    out, problems = [], []

    for qid, split, question, expected, groups in ANSWERABLE:
        group_pages = []
        for group in groups:
            found = sorted({f"{d}-p{pg}" for d, pg, t in squashed for s in group if squash(s) in t},
                           key=lambda x: (x.split("-p")[0], int(x.split("-p")[1])))
            if not found:
                problems.append(f"{qid}: no page contains any of {group}")
            group_pages.append(found)
        for i, extra in enumerate(EXTRA_PAGES.get(qid, [])):
            group_pages[i] = sorted(set(group_pages[i]) | set(extra),
                                    key=lambda x: (x.split("-p")[0], int(x.split("-p")[1])))
        all_pages = sorted({p for g in group_pages for p in g},
                           key=lambda x: (x.split("-p")[0], int(x.split("-p")[1])))
        out.append({"id": qid, "split": split, "question": question, "expected": expected,
                    "answerable": True, "pages": all_pages, "page_groups": group_pages,
                    "evidence": groups})

    for qid, split, question in UNANSWERABLE:
        out.append({"id": qid, "split": split, "question": question,
                    "expected": "Not covered by these publications.", "answerable": False,
                    "pages": [], "page_groups": [], "evidence": []})

    if problems:
        raise SystemExit("evidence not found:\n  " + "\n  ".join(problems))
    path = Path("eval/questions.jsonl")
    path.write_text("".join(json.dumps(q) + "\n" for q in out), encoding="utf-8")
    print(f"wrote {len(out)} questions to {path}")
    for q in out:
        if q["answerable"]:
            print(f"  {q['id']} [{q['split']}] {len(q['page_groups'])} group(s): {q['page_groups']}")


if __name__ == "__main__":
    main()
