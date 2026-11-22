from dataclasses import replace

from .pipeline import Draft, StagePlan, Step

BND = StagePlan(
    drafts=(
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
    ),
    steps=(
        Step(rewrites=(('gpt-6.1-sol', 'high'),), focus=('materials', 'shape')),
        Step(
            rewrites=(('gpt-6-astra', 'high'),),
            focus=('materials', 'shape'),
            fallback=('gpt-6.1-sol', 'xhigh'),
        ),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high')),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high')),
        Step(rewrites=(('gpt-6-astra', 'high'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'max'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6-astra', 'high'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
    ),
    draft_keep=1,
    fill=0.999,
    render_cap=76,
)

def _sisters(plan: StagePlan) -> StagePlan:
    steps = list(plan.steps[:10])
    steps[0] = replace(steps[0], rewrites=(('gpt-6.1-sol', 'xhigh'),) * 3, judged=False, choose=True)
    for index in (1, 2, 3):
        steps[index] = replace(steps[index], judged=False)
    return replace(plan, steps=tuple(steps), final_pick=('gpt-6.1-terra', 'medium'))


PLANS = {"s3": _sisters(BND)}
ACTIVE = "s3"
ACTIVE_PLAN = PLANS[ACTIVE]
