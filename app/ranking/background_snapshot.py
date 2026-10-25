"""Исходные кандидаты и точный запрос для восстановления того же ranking run."""
from dataclasses import asdict
from datetime import datetime
import json

from app.db.news_candidates import CandidateSelectionResult, NewsCandidate
from app.ranking.event_evaluator import EventRankingModelRequest
from app.ranking.event_formula_pipeline import EventAudienceMetrics


def encode_snapshot(selection, request, audience_metrics):
    return json.dumps({'selection':asdict(selection),'request':asdict(request),
                       'audience_metrics':[asdict(v) for v in audience_metrics]}, default=str)


def decode_snapshot(snapshot):
    selection = snapshot['selection']
    candidates = []
    for item in selection['candidates']:
        item = dict(item)
        item['source_published_at'] = datetime.fromisoformat(item['source_published_at'])
        item['categories'] = tuple(item.get('categories', ()))
        candidates.append(NewsCandidate(**item))
    return (CandidateSelectionResult(window_start=datetime.fromisoformat(selection['window_start']),
        window_end=datetime.fromisoformat(selection['window_end']),window_hours=selection['window_hours'],
        candidates=tuple(candidates), published_history=tuple(selection.get('published_history', ()))), EventRankingModelRequest(**snapshot['request']),
        tuple(EventAudienceMetrics(**v) for v in snapshot['audience_metrics']))
