"""
LLM prompt builders for OASIS Agent Profile Generator
"""

import json
from typing import Dict, Any


def get_system_prompt(is_individual: bool) -> str:
    """Get system prompt based on entity type"""
    if is_individual:
        return (
            "You are an expert character writer creating social media personas for a "
            "multi-agent simulation. Your personas must feel like REAL people — messy, "
            "opinionated, contradictory, specific. Avoid generic corporate-speak or "
            "balanced-sounding descriptions. Every person has biases, blind spots, and "
            "strong feelings about something. Lean into those.\n\n"
            "Return valid JSON. All string values must be plain text (no newlines, no markdown). "
            "Use English."
        )
    return (
        "You are an expert in institutional communications creating official social media "
        "account personas for a multi-agent simulation. Institutional accounts have a distinct "
        "voice — formal but not robotic, on-message but not tone-deaf. They hedge on "
        "controversies, amplify achievements, and deflect criticism with practiced diplomacy.\n\n"
        "Return valid JSON. All string values must be plain text (no newlines, no markdown). "
        "Use English."
    )


def build_individual_persona_prompt(
    entity_name: str,
    entity_type: str,
    entity_summary: str,
    entity_attributes: Dict[str, Any],
    context: str
) -> str:
    """Build detailed persona prompt for individual entities using soul.md structure"""

    attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
    context_str = context[:3000] if context else "No additional context"

    return f"""Create a persona for this person to use in a multi-platform social media simulation.

ENTITY: {entity_name} ({entity_type})
SUMMARY: {entity_summary}
ATTRIBUTES: {attrs_str}

CONTEXT (from knowledge graph and research):
{context_str}

Return JSON with these fields:

"bio": A punchy social media bio (2-3 sentences). Not a resume — a vibe. What would this person actually write in their Twitter/Reddit bio? Include their attitude, not just their job title.

"persona": A structured character specification (800-1200 words total) using EXACTLY these three sections with markdown headers:

## SOUL (Identity & Worldview)
Write this person's core identity and belief system. Include:
- Background that shaped their worldview (not just resume facts — what experiences made them think the way they do?)
- 2-3 specific, defensible opinions on the simulation topic. Be concrete: not "supports regulation" but "believes self-regulation failed because of [specific reason], points to [specific evidence]"
- At least one named contradiction: "believes X but also Y" — real people hold contradictory views. Example: "champions free markets but quietly supports agricultural subsidies because they grew up on a farm"
- What would change their mind — what specific evidence or argument could shift their position? Or name the topic where they are genuinely unmovable and why

## STYLE (Voice & Writing Patterns)
Define how this person writes — distinctively enough that you could identify them from an anonymous post. Include:
- Sentence structure: short punchy fragments? long flowing paragraphs? rhetorical questions?
- Punctuation habits: em dashes, ellipses, ALL CAPS for emphasis, lowercase everything, excessive exclamation marks?
- Tone: sarcastic, earnest, dry, combative, professorial, folksy, techno-optimist?
- Vocabulary level: academic jargon, plain spoken, internet slang, industry buzzwords?
- Platform-specific patterns: On Twitter — do they thread or one-liner? On Reddit — do they write essays or quick takes? Do they use data/links or argue from personal experience?
- Rhetorical patterns: do they steel-man opponents? do they dunk? do they hedge? do they use analogies or go straight to data?

## BEHAVIOR (Operating Modes)
Define how this person acts on social media — their engagement personality:
- Post frequency tendency: prolific poster or occasional commenter?
- Engagement style: reply-heavy (loves arguing in threads) vs. broadcast (posts takes, rarely engages replies) vs. lurker-who-occasionally-erupts
- What triggers them to engage: do they respond to controversy? agreement? questions? misinformation? personal attacks?
- How they handle disagreement: block, mute, argue back, write a 20-tweet thread, get sarcastic, go quiet?
- Cross-platform behavior: how do they shift between Twitter (punchy), Reddit (detailed), and prediction markets (analytical)?

KEY PRINCIPLE: Someone reading the SOUL section should be able to predict this person's take on a NEW topic. If they can't, you're being too vague.

"age": Integer
"gender": "male" or "female"
"mbti": MBTI type (e.g., "INTJ")
"country": Country name
"profession": Their job title or role
"interested_topics": ["topic1", "topic2", ...] (3-6 topics)

IMPORTANT: Do NOT include karma, friend_count, follower_count, or statuses_count — those are computed separately.
"""


def build_group_persona_prompt(
    entity_name: str,
    entity_type: str,
    entity_summary: str,
    entity_attributes: Dict[str, Any],
    context: str
) -> str:
    """Build detailed persona prompt for group/institutional entities using soul.md structure"""

    attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
    context_str = context[:3000] if context else "No additional context"

    return f"""Create an official social media account persona for this organization, for use in a multi-platform simulation.

ENTITY: {entity_name} ({entity_type})
SUMMARY: {entity_summary}
ATTRIBUTES: {attrs_str}

CONTEXT (from knowledge graph and research):
{context_str}

Return JSON with these fields:

"bio": The official account bio (2-3 sentences). Professional but not boring. Think real organizational Twitter bios — they have personality within institutional constraints.

"persona": A structured communications playbook (600-900 words total) using EXACTLY these three sections with markdown headers:

## SOUL (Institutional Identity & Position)
Define what this organization stands for and how it sees the world:
- Public mission and the image it projects — what does it want people to think of it?
- Official position on the simulation topic: what's the party line? How do they frame it?
- Institutional contradictions: what tension exists between stated values and actual behavior? Example: "champions transparency but routinely delays FOIA responses" or "promotes innovation but has a deeply risk-averse legal team"
- Red lines: what will this account NEVER say or do? What positions are off-brand?

## STYLE (Voice & Tone)
Define the account's distinctive communication style:
- Register: formal/accessible? Jargon-heavy or plain language?
- Person: first person plural ("we believe") or third person ("the organization maintains")?
- Personality: does it show personality or stay buttoned-up? Does it use humor? Emoji? Exclamation marks?
- Platform-specific patterns: On Twitter — press release tone or conversational? Does it use threads? On Reddit — does it do AMAs, post data, or just broadcast?
- Engagement voice: does it respond to critics? With what tone — dismissive, measured, empathetic?

## BEHAVIOR (Operating Modes)
Define how the account operates across platforms:
- Content strategy: what does it actually post? Press releases, data visualizations, opinion pieces, event promotion, community engagement?
- Engagement patterns: does it engage in debates or just broadcast? Does it reply to individual users or only to other institutions?
- Controversy handling: ignore, deflect, address head-on, or issue carefully worded non-responses?
- Frequency and timing: prolific daily poster or occasional announcements?
- Cross-platform behavior: same voice everywhere or adapted per platform (formal on Twitter, engaged on Reddit)?

"age": 30
"gender": "other"
"mbti": MBTI type reflecting the account's communication style. VARY THIS — not all orgs are ISTJ. \
Examples: "ISTJ" (conservative, by-the-book), "ENTJ" (assertive, agenda-setting), "ENFJ" (community-building, outreach), \
"INTP" (technical, research-focused), "ESTP" (bold, action-oriented)
"country": Country where headquartered
"profession": Brief description of institutional function
"interested_topics": ["topic1", "topic2", ...] (3-6 focus areas)

IMPORTANT: Do NOT include karma, friend_count, follower_count, or statuses_count — those are computed separately.
"""
