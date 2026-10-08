"""Streamlit client for Laya decisions served by Ollaya."""

import json

import requests
import streamlit as st

API_URL = "http://localhost:11435/api/decide"

# Official CLI presets (ollaya run --preset), sent as /api/decide questions.
# Source: Ollaya docs (triage, email, guard, moderation, router) and the
# matching question tables in laya-core preset_tables.rs.
PRESETS = {
    "triage": {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "billing_question": "a question about an invoice, plan or payment method",
                "information": "general information, pricing or how-to",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure or a deadline?",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": [
                "calm and neutral",
                "concerned but civil",
                "clearly annoyed",
                "very angry or using strong language",
            ],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?",
        },
    },
    "email": {
        "category": {
            "type": "choice",
            "instructions": "Which team should handle the email in `body`?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, integrations",
                "sales": "pricing, demos, new purchases",
                "security": "phishing, scams, account compromise",
                "hr": "hiring, leave, payroll",
                "other": "none of the above",
            },
        },
        "is_spam": {
            "type": "noul",
            "instructions": "Is this email unsolicited spam or bulk marketing?",
        },
        "is_phishing": {
            "type": "noul",
            "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
            "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request in `body`?",
            "criteria": [
                "no time pressure",
                "needs attention soon",
                "blocking issue or hard deadline",
            ],
        },
        "needs_reply": {
            "type": "noul",
            "instructions": "Does the sender expect a reply?",
        },
    },
    "guard": {
        "jailbreak": {
            "type": "noul",
            "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
        },
        "prompt_injection": {
            "type": "noul",
            "instructions": "Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?",
        },
        "sensitive_data": {
            "type": "noul",
            "instructions": "Does `prompt` contain credentials, personal data or other sensitive information?",
        },
        "harm_severity": {
            "type": "score",
            "instructions": "How much harm would complying with `prompt` cause?",
            "criteria": [
                "none: ordinary request",
                "minor: mildly inappropriate",
                "serious: unsafe advice or abuse",
                "severe: dangerous or illegal",
            ],
        },
        "topic": {
            "type": "choice",
            "instructions": "What is `prompt` about?",
            "criteria": {
                "product_support": "product support",
                "coding": "coding",
                "general_knowledge": "general knowledge",
                "personal_advice": "personal advice",
                "security_testing": "security testing",
                "other": "other",
            },
        },
    },
    "moderation": {
        "toxic": {
            "type": "noul",
            "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
        },
        "harassment": {
            "type": "noul",
            "instructions": "Does `post` target or harass a specific person?",
        },
        "threat": {
            "type": "noul",
            "instructions": "Does `post` threaten violence, harm or intimidation?",
        },
        "spam": {
            "type": "noul",
            "instructions": "Is `post` spam or advertising?",
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is any rule-breaking in `post`?",
            "criteria": [
                "no rule-breaking: ordinary on-topic post",
                "mild: rude tone or off-topic, no target",
                "clear violation: insults, harassment or spam aimed at someone",
                "severe: threats, hate speech or calls for violence",
            ],
        },
    },
    "router": {
        "difficulty": {
            "type": "score",
            "instructions": "How hard is `request` for a language model?",
            "criteria": [
                "trivial: a lookup or one-liner",
                "easy: short answer, no reasoning",
                "moderate: several steps",
                "hard: long multi-step reasoning or specialist knowledge",
            ],
        },
        "domain": {
            "type": "choice",
            "instructions": "What domain does `request` belong to?",
            "criteria": {
                "code": "software engineering, programming, refactoring, architecture, debugging",
                "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
                "writing": "creative writing, essays, emails, blog posts, copywriting",
                "factual_lookup": "facts, definitions, trivia, history",
                "data_analysis": "statistics, SQL, data manipulation, metrics",
                "chitchat": "casual conversation, greetings, small talk",
            },
        },
        "needs_tools": {
            "type": "noul",
            "instructions": "Does answering `request` require external tools, search or private data?",
        },
        "is_sensitive": {
            "type": "noul",
            "instructions": "Does `request` involve money, legal, medical or safety consequences?",
        },
    },
}

CUSTOM_TEMPLATE = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "invoice": "Needs an invoice or receipt",
            "refund": "Wants money back",
            "other": "Anything else",
        },
    }
}


st.set_page_config(page_title="Laya", layout="centered")
st.title("Laya")
st.caption("Typed decisions from the local Ollaya server.")

text = st.text_area(
    "Text",
    height=180,
    placeholder="Paste the ticket, email, or prompt you want classified.",
)
preset = st.selectbox(
    "Preset",
    ["triage", "email", "guard", "moderation", "router", "custom"],
)

questions = PRESETS.get(preset)
questions_ok = True
if preset == "custom":
    raw_questions = st.text_area(
        "Questions",
        value=json.dumps(CUSTOM_TEMPLATE, indent=2),
        height=240,
        help="Question id to a choice, score, or noul object. This is the questions field on POST /api/decide.",
    )
    try:
        questions = json.loads(raw_questions)
        if not isinstance(questions, dict) or not questions:
            questions_ok = False
            st.error("Questions must be a non-empty JSON object.")
    except json.JSONDecodeError as exc:
        questions_ok = False
        questions = None
        st.error(f"Questions JSON is invalid: {exc}")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.error("Enter the text you want to classify.")
    elif not questions_ok or questions is None:
        st.error("Fix the questions JSON before analyzing.")
    else:
        payload = {
            "model": "laya",
            "state": text,
            "questions": questions,
        }
        try:
            with st.spinner("Asking Laya..."):
                response = requests.post(API_URL, json=payload, timeout=300)
        except requests.RequestException as exc:
            st.error(
                f"Ollaya is unreachable at {API_URL}. "
                f"Start the ollaya service and keep the SSH tunnel open. ({exc})"
            )
        else:
            try:
                body = response.json()
            except ValueError:
                body = None
            if response.status_code == 200 and body is not None:
                st.json(body)
            elif body is not None:
                message = body.get("error") if isinstance(body, dict) else None
                st.error(message or f"Ollaya returned HTTP {response.status_code}.")
                st.json(body)
            else:
                st.error(f"Ollaya returned HTTP {response.status_code} with a non-JSON body.")
                st.code(response.text)
