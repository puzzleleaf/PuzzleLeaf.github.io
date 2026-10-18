// Scripted development cases. No real model, CLI, file or server is contacted.
export {};
function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text = '') {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
function button(text: string, className = '') { const node = el('button', className, text); node.type = 'button'; return node; }
function select(label: string, items: [string, string][]) {
  const node = el('select'); node.setAttribute('aria-label', label);
  items.forEach(([value, text]) => { const option = el('option', '', text); option.value = value; node.append(option); });
  return node;
}
function checkbox(text: string, checked = true) {
  const input = el('input'); input.type = 'checkbox'; input.checked = checked;
  const label = el('label', 'story-check'); label.append(input, el('span', '', text)); return { input, label };
}
const drawings: Record<string, string> = {
  person: '<circle cx="32" cy="18" r="10"/><path d="M14 54v-9c0-18 36-18 36 0v9M27 20q5 4 10 0M16 54h32"/>',
  pi: '<path d="M16 7l32 2-2 47-32-2zM21 8l-2 47M11 18h12M10 30h12M9 43h12M29 38q5 5 11 0"/><circle cx="29" cy="27" r="1"/><circle cx="40" cy="28" r="1"/>',
  model: '<path d="M16 20q0-16 16-10 15-6 17 10 13 3 8 17-2 10-16 9H21Q3 47 7 32q-2-10 9-12zM27 34q5 5 11 0M23 53h20M53 5v8M49 9h8"/><circle cx="23" cy="26" r="1"/><circle cx="41" cy="26" r="1"/>',
  bridge: '<path d="M24 6v15M40 6v15M17 21h30v9c0 10-30 10-30 0zM32 38v8q0 10 15 10M13 49h8M17 45v8"/>',
  runner: '<path d="M7 12h50v40H7zM7 22h50M13 17h2M20 17h2M16 31l7 6-7 6M30 43h14"/>',
  file: '<path d="M15 7h24l10 11v38H15zM39 7v12h10M23 29h18M23 37h18M23 45h11"/>',
};
function icon(name: string) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 64 64'); svg.setAttribute('aria-hidden', 'true'); svg.classList.add('story-icon');
  svg.innerHTML = drawings[name]; return svg;
}
function wait(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException('Paused', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, ms);
    if (signal.aborted) abort(); else signal.addEventListener('abort', abort, { once: true });
  });
}
type Owner = 'pi' | 'app' | 'model' | 'cli';
type CodeRef = { symbol: string; path: string; from: number; to: number };
type SourceStep = { id: string; title: string; owner: Owner; source: CodeRef; detail: string; ownCode?: string };
const sourceBase = 'https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/';
const loopFile = 'packages/agent/src/agent-loop.ts';
const sourceRefs = {
  prompt: { symbol: 'AgentSession.prompt()', path: 'packages/coding-agent/src/core/agent-session.ts', from: 2058, to: 2147 },
  subscribe: { symbol: 'AgentSession.subscribe()', path: 'packages/coding-agent/src/core/agent-session.ts', from: 1373, to: 1383 },
  model: { symbol: 'streamAssistantResponse()', path: loopFile, from: 381, to: 410 },
  toolCall: { symbol: 'runLoop() · toolCall', path: loopFile, from: 258, to: 279 },
  validate: { symbol: 'prepareToolCall()', path: loopFile, from: 708, to: 780 },
  execute: { symbol: 'executePreparedToolCall()', path: loopFile, from: 821, to: 857 },
  result: { symbol: 'createToolResultMessage()', path: loopFile, from: 930, to: 949 },
  dispatch: { symbol: 'executeToolCalls()', path: loopFile, from: 508, to: 526 },
  preflight: { symbol: 'prepareToolCall() · 배치 준비', path: loopFile, from: 586, to: 620 },
  completed: { symbol: 'emitToolExecutionEnd()', path: loopFile, from: 621, to: 646 },
  ordered: { symbol: 'Promise.all() → 결과 메시지', path: loopFile, from: 646, to: 663 },
  sequentialPrepare: { symbol: 'executeToolCallsSequential() · 호출 준비', path: loopFile, from: 529, to: 566 },
  sequentialResult: { symbol: 'executeToolCallsSequential() · 완료·기록', path: loopFile, from: 567, to: 584 },
} satisfies Record<string, CodeRef>;
const ownerNames: Record<Owner, string> = { pi: 'Pi가 처리', app: '내가 구현', model: '모델이 판단', cli: '외부 도구가 처리' };
function sourceTracker(root: HTMLElement, steps: SourceStep[]) {
  const skipped = new Set<string>();
  const element = el('div', 'story-source');
  const rail = el('ol', 'story-source-rail'); rail.setAttribute('aria-label', '작업 단계');
  rail.style.gridTemplateColumns = `repeat(${Math.min(steps.length, 7)}, minmax(0, 1fr))`;
  const markers = steps.map((step, index) => {
    const marker = el('li'); marker.dataset.owner = step.owner; marker.title = `${index + 1}. ${step.title} · ${ownerNames[step.owner]}`;
    marker.append(el('span', '', String(index + 1).padStart(2, '0')), el('span', 'sr-only', `${step.title} · ${ownerNames[step.owner]}`));
    rail.append(marker); return marker;
  });
  const heading = el('div', 'story-source-heading'), owner = el('strong', 'story-owner'), title = el('span', 'story-source-title');
  heading.append(owner, title);
  const detail = el('span', 'story-source-detail'), ownCode = el('code', 'story-own-code');
  const reference = el('div', 'story-source-ref'), relation = el('span'), link = el('a'), location = el('span', 'story-source-location');
  link.target = '_blank'; link.rel = 'noopener noreferrer'; reference.append(relation, link, location);
  element.append(rail, heading, detail, ownCode, reference);
  function set(id: string, override: Partial<Pick<SourceStep, 'owner' | 'title' | 'detail' | 'ownCode' | 'source'>> = {}) {
    const index = steps.findIndex(step => step.id === id);
    if (index < 0) throw new Error(`Unknown source step: ${id}`);
    const step = { ...steps[index], ...override };
    root.dataset.sourceStep = id; root.dataset.owner = step.owner; element.dataset.owner = step.owner;
    markers.forEach((marker, i) => {
      const displayed = i === index ? step : steps[i]; marker.dataset.owner = displayed.owner;
      marker.title = `${i + 1}. ${displayed.title} · ${skipped.has(displayed.id) ? '이번 경로에서는 생략' : ownerNames[displayed.owner]}`;
      marker.querySelector('.sr-only')!.textContent = `${displayed.title} · ${ownerNames[displayed.owner]}`;
      marker.dataset.state = skipped.has(displayed.id) ? 'skipped' : i === index ? 'current' : i < index ? 'done' : 'next';
      if (i === index) marker.setAttribute('aria-current', 'step'); else marker.removeAttribute('aria-current');
    });
    owner.textContent = ownerNames[step.owner]; title.textContent = `${index + 1} / ${steps.length} · ${step.title}`;
    detail.textContent = step.detail; ownCode.textContent = step.ownCode ?? ''; ownCode.hidden = !step.ownCode;
    relation.textContent = step.owner === 'pi' ? 'Pi 구현' : step.owner === 'model' ? 'Pi가 이 응답을 받는 곳' : 'Pi 연결점';
    link.textContent = step.source.symbol;
    link.href = `${sourceBase}${step.source.path}#L${step.source.from}-L${step.source.to}`;
    location.textContent = `${step.source.path.split('/').at(-1)} · ${step.source.from}–${step.source.to}`;
  }
  const pauseAtSource = () => root.querySelector<HTMLButtonElement>('.story-controls button:nth-child(2)')?.click();
  link.addEventListener('focus', pauseAtSource); link.addEventListener('click', pauseAtSource);
  set(steps[0].id);
  return { element, set, skip: (ids: string[]) => { skipped.clear(); ids.forEach(id => skipped.add(id)); } };
}
function shell(root: HTMLElement, title: string, visibilityTarget: Element = root) {
  const header = el('div', 'story-header'); header.append(el('strong', '', title));
  const options = el('div', 'story-options'), stage = el('div', 'story-stage'), controls = el('div', 'story-controls');
  const play = button('장면 재생', 'story-play'), pause = button('멈춤'), reset = button('처음 장면'); pause.disabled = true;
  const status = el('span', 'story-status', '준비'); status.setAttribute('role', 'status');
  controls.append(play, pause, reset, status);
  const mount = root.querySelector<HTMLElement>('[data-lab-mount]')!;
  mount.replaceChildren(header, options, controls, stage);
  let current: AbortController | undefined;
  let repeatTimer: ReturnType<typeof setTimeout> | undefined;
  let wantsPlayback = true, inView = false;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const article = root.closest<HTMLElement>('[data-article-panel]');
  function canRepeat() { return wantsPlayback && inView && !document.hidden && !article?.hidden && !reducedMotion.matches; }
  function clearRepeat() { clearTimeout(repeatTimer); repeatTimer = undefined; }
  function updateControls() {
    play.disabled = Boolean(current);
    pause.disabled = !current && repeatTimer === undefined;
    root.dataset.running = String(Boolean(current));
    root.dataset.autoplay = String(wantsPlayback && !reducedMotion.matches);
  }
  function schedule(delay = 0) {
    clearRepeat();
    if (canRepeat() && !current) {
      repeatTimer = setTimeout(() => {
        repeatTimer = undefined;
        if (canRepeat()) play.click();
        updateControls();
      }, delay);
    }
    updateControls();
  }
  function stop(manual = true) {
    if (manual) wantsPlayback = false;
    clearRepeat(); current?.abort(); current = undefined;
    status.setAttribute('aria-live', manual ? 'polite' : 'off');
    status.textContent = manual ? '멈춤' : '준비';
    root.dispatchEvent(new Event('story-stop')); updateControls();
  }
  function restart(resetScene: () => void) {
    stop(false); resetScene(); schedule(350);
  }
  async function run(action: (signal: AbortSignal) => Promise<void>) {
    if (current) return;
    clearRepeat();
    const controller = new AbortController(); current = controller; updateControls();
    root.dataset.cycle = String(Number(root.dataset.cycle ?? 0) + 1);
    const live = canRepeat() ? 'off' : 'polite';
    status.setAttribute('aria-live', live);
    stage.querySelectorAll('[aria-live]').forEach(node => node.setAttribute('aria-live', live));
    status.textContent = '재생 중';
    try {
      await action(controller.signal);
      if (current === controller) status.textContent = canRepeat() ? '잠시 후 반복' : '끝';
    } catch (error) {
      if (current === controller) {
        wantsPlayback = false;
        status.textContent = error instanceof DOMException && error.name === 'AbortError' ? '멈춤' : '오류';
      }
    } finally {
      if (current === controller) { current = undefined; schedule(2000); }
    }
  }
  play.addEventListener('click', () => { wantsPlayback = true; clearRepeat(); });
  pause.addEventListener('click', () => stop());
  new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting && entries[0].intersectionRatio >= .2;
    if (inView) schedule(); else stop(false);
  }, { threshold: [0, .2] }).observe(visibilityTarget);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(false); else schedule(); });
  if (article) new MutationObserver(() => { if (article.hidden) stop(false); else schedule(); }).observe(article, { attributes: true, attributeFilter: ['hidden'] });
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) stop(false); else schedule(); });
  updateControls();
  root.dataset.ready = 'true';
  return { options, stage, play, reset, status, run, stop, restart };
}

function actors() {
  const scene = el('div', 'story-cast');
  const links = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  links.classList.add('story-links'); links.setAttribute('viewBox', '0 0 600 260'); links.setAttribute('preserveAspectRatio', 'none'); links.setAttribute('aria-hidden', 'true');
  links.innerHTML = '<path data-edge="request" d="M155 64H245"/><path data-edge="model" d="M355 64H445"/><path data-edge="tool" d="M300 111V163"/><path data-edge="cli" d="M355 195H445"/>';
  scene.append(links);
  const nodes = new Map<string, HTMLElement>();
  [['person', '개발자', '작업 요청'], ['pi', 'Pi 하네스', 'Pi 제공'], ['model', '판단 모델', '모델의 판단'], ['runner', 'CLI 연결 코드', '직접 구현'], ['cli', 'Claude Code', '도구 자체 실행']].forEach(([id, title, role]) => {
    const actor = el('div', `story-actor story-actor--${id}`); actor.append(icon(id === 'runner' ? 'bridge' : id === 'cli' ? 'runner' : id), el('strong', '', title), el('small', '', role)); scene.append(actor); nodes.set(id, actor);
  });
  const speech = el('div', 'story-speech'); const speaker = el('small'), message = el('p'); speech.append(speaker, message);
  speech.setAttribute('aria-live', 'polite'); speech.setAttribute('aria-atomic', 'true');
  function say(who: string, text: string, edge = '', reverse = false) {
    nodes.forEach((node, id) => node.dataset.active = String(id === who));
    links.querySelectorAll('path').forEach(path => { path.classList.toggle('is-moving', path.getAttribute('data-edge') === edge); path.classList.toggle('is-reverse', reverse); });
    speaker.textContent = nodes.get(who)?.querySelector('strong')?.textContent ?? ''; message.textContent = text; scene.dataset.actor = who;
  }
  function settle() { links.querySelectorAll('path').forEach(path => path.classList.remove('is-moving')); }
  function setReviewer(name: string) { nodes.get('cli')!.querySelector('strong')!.textContent = name; }
  return { scene, speech, say, settle, setReviewer };
}
function featurePreview() {
  const element = el('div', 'feature-preview'), heading = el('small', 'feature-preview-label', '개발할 기능 · 검색 결과의 태그 필터');
  const form = el('div', 'feature-query-row'), queryLabel = el('span'), tagLabel = el('span', 'feature-selected-tag'); form.append(queryLabel, tagLabel);
  const results = el('ul', 'feature-results'), state = el('span', 'feature-preview-state'); element.append(heading, form, results, state);
  const posts = [
    { title: 'Pi 하네스 읽기', tag: 'AI' },
    { title: '검색 도우미 만들기', tag: 'AI' },
    { title: 'Astro 검색 정리', tag: 'Astro' },
  ];
  function show(query: string, tag: string, message: string) {
    queryLabel.textContent = `검색어: ${query}`; tagLabel.textContent = `태그: ${tag === 'all' ? '전체' : tag}`;
    const matches = posts.filter(post => post.title.includes(query) && (tag === 'all' || post.tag === tag));
    results.replaceChildren(...matches.map(post => {
      const item = el('li'); item.append(el('span', '', post.title), el('small', '', post.tag)); return item;
    }));
    if (!matches.length) results.append(el('li', 'feature-empty', '일치하는 글이 없습니다.'));
    state.textContent = message; element.dataset.tag = tag; element.dataset.query = query;
  }
  show('Pi', 'all', '요구사항: 검색어와 태그를 함께 적용하고, 검색어를 바꿔도 태그 선택을 유지합니다.');
  return { element, show };
}
function initReview(root: HTMLElement) {
  const app = shell(root, '사례 1 · Codex 구현 → Claude 리뷰 → 보완');
  app.stage.classList.add('story-stage--feature');
  const outcome = select('개발 진행 상황', [['needs-work', 'Claude가 보완점을 찾음'], ['clean', '첫 구현이 리뷰를 통과함'], ['failed', 'Claude 실행이 실패함']]);
  app.options.append(outcome);
  const change = el('div', 'story-request', '개발할 기능 · 블로그 검색에 태그 필터 추가');
  change.append(el('code', 'story-snapshot', '구현: Codex · 리뷰: Claude Code · 최종 반영: 개발자'));
  const preview = featurePreview();
  const milestones = el('div', 'feature-milestones');
  const milestoneItems = ['Codex 구현', 'Claude 리뷰', 'Codex 보완', '검사·완료'].map(title => {
    const item = el('span', '', title); milestones.append(item); return item;
  });
  const cast = actors(), receipt = el('div', 'story-receipt'); receipt.hidden = true;
  const source = sourceTracker(root, [
    { id: 'request', title: '기능 요구사항 전달', owner: 'app', source: sourceRefs.prompt, ownCode: '요구사항·대상 프로젝트 → session.prompt()', detail: '목표와 완료 기준, 사용할 작업 공간은 프로젝트에서 정합니다.' },
    { id: 'choose', title: '구현 작업 선택', owner: 'model', source: sourceRefs.toolCall, detail: '판단 모델이 implement_feature 호출을 요청합니다. 구현자는 Codex로 정한 사례입니다.' },
    { id: 'validate', title: '구현 도구 검증', owner: 'pi', source: sourceRefs.validate, detail: 'Pi가 등록된 도구와 입력을 검사합니다. 작업 순서·권한 정책은 직접 작성합니다.' },
    { id: 'execute', title: 'Codex 실행 연결', owner: 'app', source: sourceRefs.execute, ownCode: 'implement_feature.execute() → runCodex()', detail: '내 연결 코드가 분리한 작업 공간에서 Codex를 실행합니다.' },
    { id: 'implement', title: '태그 필터 구현', owner: 'cli', source: sourceRefs.execute, detail: 'Codex가 검색 UI와 필터 처리를 구현합니다. 실제 파일 수정은 Codex의 하네스가 맡습니다.' },
    { id: 'implementation-result', title: '구현 결과 연결', owner: 'pi', source: sourceRefs.result, detail: 'Pi가 구현 보고서를 도구 결과로 기록해 다음 모델 요청에 연결합니다. 변경 버전 식별자는 앱이 만듭니다.' },
    { id: 'review-bridge', title: 'Claude 리뷰 연결', owner: 'app', source: sourceRefs.execute, ownCode: 'review_change.execute() → runClaudeReview()', detail: '모델이 review_change를 요청하고 Pi의 검증을 거치면, 내 코드가 v1과 완료 기준을 Claude에 전달합니다.' },
    { id: 'review', title: '기능 요구사항 리뷰', owner: 'cli', source: sourceRefs.execute, detail: 'Claude Code가 검색어 변경 뒤에도 태그 선택이 유지되는지 검토합니다. 리뷰 내용은 Pi가 작성하지 않습니다.' },
    { id: 'review-result', title: '리뷰 결과 연결', owner: 'pi', source: sourceRefs.result, detail: 'Pi가 지적 사항 또는 실행 실패를 도구 결과로 이어 줍니다. 지적 발견과 실행 실패는 다릅니다.' },
    { id: 'decide', title: '보완 여부 판단', owner: 'model', source: sourceRefs.toolCall, detail: '모델이 리뷰 결과를 보고 다음 도구 호출을 고릅니다. 보완 횟수와 허용 순서는 앱 정책으로 제한합니다.' },
    { id: 'revise', title: 'Codex가 보완', owner: 'cli', source: sourceRefs.execute, detail: '리뷰 지적을 받은 Codex가 검색어 변경 시 태그가 초기화되는 문제를 고칩니다. Pi는 도구 호출을 이어 줍니다.' },
    { id: 'review-again', title: '새 버전 재리뷰', owner: 'cli', source: sourceRefs.execute, detail: '내 코드가 새 변경 버전을 고정해 Claude Code에 다시 넘깁니다. 이전 버전의 리뷰와 섞지 않습니다.' },
    { id: 'checks', title: '빌드·동작 검사', owner: 'app', source: sourceRefs.execute, ownCode: 'run_checks.execute() → 프로젝트 검사 스크립트', detail: '정해 둔 빌드·검색 조건 검사를 내 코드가 실행합니다. Pi가 검사 항목을 자동으로 만들어 주지는 않습니다.' },
    { id: 'display', title: '개발자에게 결과 전달', owner: 'app', source: sourceRefs.subscribe, ownCode: 'session.subscribe() → 상태 갱신·결과 표시', detail: '내 앱이 Pi의 이벤트를 받아 결과를 보여 줍니다. 실제 반영·커밋·배포 여부는 별도로 결정합니다.' },
  ]);
  cast.speech.append(source.element);
  root.addEventListener('story-stop', cast.settle);
  app.stage.append(change, milestones, preview.element, cast.scene, cast.speech, receipt);
  function milestone(index: number) {
    milestoneItems.forEach((item, i) => {
      const skip = i === 2 && index === 3 && outcome.value === 'clean';
      item.dataset.state = skip ? 'skipped' : i === index ? 'current' : i < index ? 'done' : 'next';
      item.textContent = skip ? '보완 생략' : ['Codex 구현', 'Claude 리뷰', 'Codex 보완', '검사·완료'][i];
    });
    root.dataset.workflowPhase = ['implement', 'review', 'revise', 'checks'][index];
  }
  function initial() {
    source.skip(outcome.value === 'clean' ? ['revise', 'review-again'] : outcome.value === 'failed' ? ['revise', 'review-again', 'checks'] : []);
    source.set('request'); milestone(0); cast.setReviewer('Codex');
    cast.say('person', '“검색에 태그 필터를 넣어 줘. Codex가 구현하고 Claude가 리뷰해 줘.”');
    preview.show('Pi', 'all', '완료 기준: 검색어와 태그를 함께 적용 · 검색어 변경 시 태그 유지 · 빈 결과 안내');
    cast.settle(); receipt.hidden = true; delete root.dataset.result;
  }
  initial();
  app.play.addEventListener('click', () => {
    const result = outcome.value, needsWork = result === 'needs-work', failed = result === 'failed';
    void app.run(async signal => {
      initial();
      async function step(id: string, who: string, text: string, edge = '', reverse = false) {
        source.set(id, id === 'review' && failed ? { owner: 'app', title: '리뷰 실행 실패 확인', ownCode: 'runClaudeReview(): 실행 상태·오류 확인', detail: '내 연결 코드가 Claude 실행 실패를 확인합니다. 리뷰가 끝난 것으로 처리하지 않습니다.' } : {});
        cast.say(who, text, edge, reverse); root.dataset.frame = id;
        await wait(1050, signal);
      }
      await step('request', 'person', '“검색어와 태그를 함께 적용하고, 검색어를 바꿔도 선택한 태그를 유지해 줘.”', 'request');
      await step('choose', 'model', 'implement_feature를 요청합니다. 구현은 Codex에 맡깁니다.', 'model', true);
      await step('validate', 'pi', '구현 도구와 입력을 확인합니다. 검증한 뒤 등록된 함수를 부릅니다.', 'tool');
      await step('execute', 'runner', 'Codex에 요구사항과 작업 공간을 전달합니다.', 'tool');
      preview.show('Pi', 'AI', 'Codex 초안 · 검색에 태그 선택 UI와 필터 조건을 추가했습니다.');
      await step('implement', 'cli', '태그 선택 UI와 검색어·태그를 함께 적용하는 처리를 구현했습니다.', 'cli', true);
      await step('implementation-result', 'pi', '구현 보고서와 변경 버전 v1을 다음 판단에 연결합니다.', 'model');
      milestone(1); cast.setReviewer('Claude Code');
      await step('review-bridge', 'runner', 'Claude Code에 v1 변경 내용과 완료 기준을 전달합니다.', 'tool');
      if (!failed) preview.show('검색', needsWork ? 'all' : 'AI', needsWork ? 'Claude 리뷰 · 검색어를 바꾸자 선택했던 AI 태그가 풀렸습니다.' : 'Claude 리뷰 · 검색어가 바뀌어도 AI 태그가 유지됩니다.');
      await step('review', failed ? 'runner' : 'cli', failed ? 'Claude 실행 실패 · 리뷰 결과를 받지 못했습니다.' : needsWork ? '보완점: 검색어를 바꿀 때 selectedTag가 초기화됩니다. 태그 선택을 유지해 주세요.' : '리뷰 완료 · 검색어 변경 뒤 태그 유지와 빈 결과 처리에서 추가 지적이 없습니다.', failed ? 'tool' : 'cli', true);
      await step('review-result', 'pi', failed ? '리뷰 실행 실패를 전달합니다. 리뷰를 통과한 것으로 처리하지 않습니다.' : needsWork ? 'Claude의 지적 사항을 도구 결과로 다음 모델 요청에 붙입니다.' : 'Claude의 리뷰 완료 결과를 다음 판단으로 이어 줍니다.', 'model');
      await step('decide', 'model', failed ? '리뷰를 마치지 못했으므로 자동 보완을 이어 가지 않고 상태를 보고합니다.' : needsWork ? '지적을 붙여 implement_feature를 다시 요청합니다. 이번 사례에서는 보완을 한 번만 허용합니다.' : '보완은 건너뛰고 run_checks로 검사합니다.', 'model', true);
      if (needsWork) {
        milestone(2); cast.setReviewer('Codex');
        preview.show('검색', 'AI', 'Codex 보완안 v2 · 검색어가 바뀌어도 선택한 AI 태그를 유지합니다.');
        await step('revise', 'cli', 'Claude의 지적을 반영해 태그 선택 상태를 유지하도록 보완했습니다.', 'cli', true);
        cast.setReviewer('Claude Code');
        await step('review-again', 'cli', 'v2 재리뷰 완료 · 태그 선택이 유지되고 빈 결과 안내도 확인했습니다.', 'cli', true);
      }
      if (!failed) {
        milestone(3);
        preview.show('없는 제목', 'AI', '동작 검사 · 검색어와 태그에 맞는 글이 없을 때 안내를 보여 줍니다.');
        await step('checks', 'runner', '빌드와 검색 조건 검사를 실행합니다. 선택 태그 유지·빈 결과 안내를 확인합니다.', 'tool');
        preview.show('검색', 'AI', needsWork ? '보완안 v2 · 선택 태그 유지와 빈 결과 안내까지 확인했습니다.' : '구현안 v1 · 선택 태그 유지와 빈 결과 안내까지 확인했습니다.');
      }
      const final = failed ? '리뷰 미완료 · 구현 결과를 보관하고 실행 상태 확인' : needsWork ? 'Codex 구현 → Claude 지적 → Codex 보완 → 재리뷰·검사 완료' : 'Codex 구현 → Claude 리뷰 → 검사 완료';
      await step('display', 'person', failed ? '리뷰를 완료하지 못했으므로 최종 반영은 보류합니다.' : '변경 내용, Claude 리뷰, 검사 결과를 함께 보고 최종 반영을 결정합니다.', 'request', true);
      cast.settle(); root.dataset.result = result;
      receipt.replaceChildren(icon('file'), el('strong', '', failed ? '리뷰 실행 실패' : '기능 개발 흐름 완료'), el('span', '', final)); receipt.hidden = false;
    });
  });
  app.reset.addEventListener('click', () => { app.stop(); initial(); app.status.textContent = '준비'; });
  outcome.addEventListener('change', () => app.restart(initial));
}

function initHandoff(root: HTMLElement) {
  const app = shell(root, '사례 2 · Codex 구현 결과를 Claude에게 인계');
  const include = checkbox('검토 기준 함께 전달'), display = checkbox('작업 명세 펼쳐 보기');
  app.options.append(include.label, display.label);
  const scene = el('div', 'story-memory');
  const storage = el('div', 'story-note'), memo = el('p'), folded = el('span', 'story-folded', '접어 둔 작업 명세');
  storage.append(el('small', '', 'Codex 구현 결과 v1'), icon('file'), memo, folded);
  const gate = el('div', 'story-gate', '→'), desk = el('div', 'story-note story-desk');
  root.addEventListener('story-stop', () => gate.classList.remove('is-moving'));
  const sent = el('p'), destination = el('small'); desk.append(destination, sent); scene.append(storage, gate, desk);
  const speech = el('div', 'story-speech'), who = el('small'), words = el('p'); speech.append(who, words);
  speech.setAttribute('aria-live', 'polite'); speech.setAttribute('aria-atomic', 'true');
  const source = sourceTracker(root, [
    { id: 'packet', title: '구현 버전·기준 선택', owner: 'app', source: sourceRefs.execute, ownCode: 'runClaudeReview(): 구현 버전 + 완료 기준 선택', detail: 'Codex의 구현 결과와 요구사항을 내 코드가 묶습니다. Pi 기록이 CLI로 자동 복사되지는 않습니다.' },
    { id: 'handoff', title: 'Claude에 리뷰 입력 전달', owner: 'app', source: sourceRefs.execute, ownCode: 'runClaudeReview(): CLI 인자·표준 입력 구성', detail: '리뷰할 변경 버전을 고정하고 완료 기준을 함께 전달합니다. Pi 링크는 이 코드를 부르는 연결점입니다.' },
    { id: 'external', title: '요구사항 리뷰 시작', owner: 'cli', source: sourceRefs.execute, detail: 'Claude Code가 Codex의 구현과 완료 기준을 대조합니다. 이 내부 검토 구현은 Pi 저장소에 없습니다.' },
  ]);
  speech.append(source.element);
  const receipt = el('div', 'story-receipt'); receipt.hidden = true;
  app.stage.append(scene, speech, receipt);
  const criteria = '완료 기준: 검색어와 태그 함께 적용 · 검색어 변경 시 태그 유지 · 빈 결과 안내';
  function update() {
    source.set('packet');
    memo.textContent = `기능: 검색 태그 필터\n구현 버전: search-tag-v1\n${criteria}`;
    memo.hidden = !display.input.checked; folded.hidden = display.input.checked;
    destination.textContent = 'Claude Code에 넘길 리뷰 입력'; sent.textContent = '구현 버전 v1 전달 준비';
    gate.classList.remove('is-moving'); receipt.hidden = true;
    who.textContent = '개발자'; words.textContent = '“Codex가 만든 기능이 요구사항을 만족하는지 Claude에게 리뷰 맡겨 줘.”';
  }
  update();
  app.play.addEventListener('click', () => {
    const keep = include.input.checked;
    void app.run(async signal => {
      update(); await wait(900, signal); source.set('handoff');
      gate.classList.add('is-moving'); sent.textContent = keep ? `구현 버전: search-tag-v1\n${criteria}` : '구현 버전: search-tag-v1';
      who.textContent = '내 리뷰 연결 코드'; words.textContent = `Claude Code에 ${keep ? '구현 결과와 완료 기준을' : '구현 결과만'} 전달합니다.`;
      await wait(1100, signal); gate.classList.remove('is-moving'); who.textContent = 'Claude Code'; source.set('external');
      words.textContent = keep ? '“검색어를 바꿔도 태그가 유지되는지, 결과가 없을 때 안내하는지 확인하겠습니다.”' : '“태그 선택을 유지해야 하나요? 리뷰할 완료 기준도 알려 주세요.”';
      receipt.replaceChildren(icon('file'), el('strong', '', keep ? '완료 기준까지 인계됨' : '완료 기준 누락'), el('span', '', keep ? 'Claude가 구현 결과와 요구사항을 함께 받았습니다.' : '구현 코드만으로는 원하는 동작을 모두 알 수 없습니다.'));
      receipt.hidden = false;
      await wait(1000, signal);
    });
  });
  app.reset.addEventListener('click', () => { app.stop(); update(); app.status.textContent = '준비'; });
  include.input.addEventListener('change', () => app.restart(update));
  display.input.addEventListener('change', () => { memo.hidden = !display.input.checked; folded.hidden = display.input.checked; });
}

function initReviewers(root: HTMLElement) {
  const app = shell(root, '사례 3 · Claude 리뷰와 프로젝트 검사를 함께');
  const mode = select('검토 실행 방식', [['parallel', '리뷰와 검사를 함께 시작'], ['sequential', '리뷰 뒤 검사를 시작']]);
  const faster = select('먼저 답하는 도구', [['checks', '프로젝트 검사가 먼저 완료'], ['claude-code', 'Claude 리뷰가 먼저 완료']]);
  app.options.append(mode, faster);
  const request = el('div', 'story-request', 'Codex 구현 완료 v1 → 같은 변경 버전으로 리뷰·검사');
  const branches = el('div', 'story-branches');
  const findings = ['검색어를 바꾸면 선택 태그가 풀립니다. 보완이 필요합니다.', '빌드는 통과했습니다.'];
  const nodes = ['Claude Code 리뷰', '프로젝트 검사'].map(name => {
    const node = el('div', 'story-branch'), state = el('span', '', '검토 전');
    node.append(icon('runner'), el('strong', '', name), state); branches.append(node); return { node, state, name };
  });
  root.addEventListener('story-stop', () => nodes.forEach(({ node, state }) => {
    if (node.dataset.state === 'waiting') { node.dataset.state = 'paused'; state.textContent = '장면 재생 멈춤'; }
  }));
  const lists = el('div', 'story-ledgers'), arrival = el('ol'), recorded = el('ol');
  const makeList = (title: string, list: HTMLOListElement) => { const panel = el('div', 'story-ledger'); panel.append(el('small', '', title), list); lists.append(panel); };
  makeList('완료 소식이 도착한 순서', arrival); makeList('Pi의 도구 결과 메시지', recorded);
  const answer = el('div', 'story-speech'); answer.append(el('small', '', '판단 모델'), el('p', '', '리뷰와 검사 결과를 기다립니다.'));
  const source = sourceTracker(root, [
    { id: 'dispatch', title: '실행 방식 결정', owner: 'pi', source: sourceRefs.dispatch, detail: 'Pi가 도구의 실행 모드를 보고 병렬 또는 순차 경로를 고릅니다.' },
    { id: 'preflight', title: '호출 준비', owner: 'pi', source: sourceRefs.preflight, detail: '병렬 경로에서는 실행 전에 도구 호출을 준비합니다. 순차 경로는 호출마다 준비합니다.' },
    { id: 'execute', title: '리뷰·검사 실행 연결', owner: 'app', source: sourceRefs.execute, ownCode: 'runClaudeReview() / runChecks()', detail: 'Claude 리뷰와 프로젝트 검사 실행은 직접 연결합니다. 두 작업은 같은 구현 버전을 확인합니다.' },
    { id: 'complete', title: '완료 소식 전달', owner: 'pi', source: sourceRefs.completed, detail: '끝난 도구부터 완료 이벤트가 나옵니다. 화면 갱신은 이벤트를 받은 내 앱이 구현합니다.' },
    { id: 'record', title: '결과 순서 정리', owner: 'pi', source: sourceRefs.ordered, detail: '병렬 경로는 Promise.all() 뒤 호출 순서대로 결과 메시지를 만듭니다.' },
    { id: 'summary', title: '개발 상태 판단', owner: 'model', source: sourceRefs.model, detail: '모델이 빌드 통과와 리뷰 지적을 함께 읽습니다. 빌드 성공만으로 기능 완료라고 판단하면 안 됩니다.' },
  ]);
  answer.append(source.element);
  app.stage.append(request, branches, lists, answer);
  function clear() { source.set('dispatch'); nodes.forEach(({ node, state }) => { node.dataset.state = 'idle'; state.textContent = '실행 전'; }); arrival.replaceChildren(); recorded.replaceChildren(); answer.querySelector('p')!.textContent = '리뷰와 검사 결과를 기다립니다.'; }
  function record(i: number) { recorded.append(el('li', '', `${nodes[i].name} · ${i ? '빌드 통과' : '태그 유지 보완 필요'}`)); }
  app.play.addEventListener('click', () => {
    const parallel = mode.value === 'parallel', fast = faster.value === 'checks' ? 1 : 0;
    void app.run(async signal => {
      clear();
      await wait(700, signal); source.set('preflight', parallel ? {} : { source: sourceRefs.sequentialPrepare, detail: '순차 경로에서는 각 호출을 준비하고 하나씩 실행합니다.' });
      await wait(700, signal); source.set('execute');
      const review = async (i: number) => {
        nodes[i].node.dataset.state = 'waiting'; nodes[i].state.textContent = i ? '빌드 검사 중…' : '기능 리뷰 중…';
        await wait(i === fast ? 650 : 1800, signal);
        nodes[i].node.dataset.state = 'done'; nodes[i].state.textContent = i ? '검사 완료' : '리뷰 완료';
        source.set('complete', parallel ? {} : { source: sourceRefs.sequentialResult, detail: '한 도구가 끝나면 완료 이벤트와 결과 메시지를 연결하고 다음 호출로 넘어갑니다.' });
        arrival.append(el('li', '', `${nodes[i].name} · ${findings[i]}`));
      };
      if (parallel) { await Promise.all([review(0), review(1)]); record(0); record(1); }
      else {
        await review(0); record(0);
        source.set('preflight', { source: sourceRefs.sequentialPrepare, detail: '첫 검토가 끝났으므로 다음 도구 호출을 준비합니다.' }); await wait(500, signal);
        source.set('execute'); await review(1); record(1);
      }
      source.set('record', parallel ? {} : { source: sourceRefs.sequentialResult, detail: '순차 경로는 호출이 끝날 때마다 결과를 기록합니다. Promise.all()을 거치지 않습니다.' }); await wait(800, signal);
      source.set('summary');
      answer.querySelector('p')!.textContent = '“빌드는 통과했지만 태그 유지 동작은 보완해야 합니다. 이 지적을 Codex에 넘겨 주세요.”';
      await wait(800, signal);
    });
  });
  app.reset.addEventListener('click', () => { app.stop(); clear(); app.status.textContent = '준비'; });
  mode.addEventListener('change', () => app.restart(clear)); faster.addEventListener('change', () => app.restart(clear));
}
function initComparison(root: HTMLElement) {
  const mount = el('div', 'compare-playback'); mount.dataset.labMount = '';
  root.querySelector('.compare-headings')!.before(mount);
  const app = shell(root, '같은 작업 · 두 실행 경로', mount);
  const next = button('다음 단계'); app.reset.after(next);
  const progress = el('div', 'compare-progress'); progress.setAttribute('aria-label', '개발 작업 진행');
  const tasks = [
    { title: 'Codex 구현', tool: 'implement_feature', runner: 'runCodex(task)', result: '태그 필터 구현 결과 v1', next: 'v1을 Claude 리뷰에 넘깁니다.' },
    { title: 'Claude 리뷰', tool: 'review_change', runner: 'runClaudeReview(v1)', result: '검색어 변경 시 태그가 풀린다는 지적', next: '지적 사항을 Codex의 보완 입력에 넣습니다.' },
    { title: 'Codex 보완', tool: 'implement_feature', runner: 'runCodex(task, review)', result: '태그 선택을 유지하도록 보완한 v2', next: '새 버전 v2를 Claude에게 다시 넘깁니다.' },
    { title: 'Claude 재리뷰', tool: 'review_change', runner: 'runClaudeReview(v2)', result: 'v2의 태그 유지 동작 리뷰 통과', next: '같은 v2의 프로젝트 검사를 요청합니다.' },
    { title: '프로젝트 검사', tool: 'run_checks', runner: 'runChecks(v2)', result: 'v2 빌드·검색 조건 검사 통과', next: '최신 버전의 리뷰와 검사를 모아 보고합니다.' },
  ];
  const milestones = tasks.map(task => { const item = el('span', '', task.title); progress.append(item); return item; });
  const speech = el('p', 'compare-live-message'); speech.setAttribute('aria-live', 'polite'); speech.setAttribute('aria-atomic', 'true');
  const pair = el('div', 'compare-live-pair');
  const makeCard = (side: string) => {
    const card = el('div', 'compare-live-card'), label = el('small'), code = el('code'), refs = el('div', 'compare-live-refs');
    card.dataset.side = side; card.append(label, code, refs); pair.append(card); return { card, label, code, refs };
  };
  const direct = makeCard('direct'), pi = makeCard('pi');
  const phase = el('div', 'compare-phase');
  app.stage.append(progress, phase, speech, pair);
  type Phase = 'request' | 'choose' | 'dispatch' | 'cli' | 'result' | 'repeat' | 'policy';
  const phases: Phase[] = ['request', 'choose', 'dispatch', 'cli', 'result', 'repeat'];
  const frames = tasks.flatMap((_, task) => phases.map(step => ({ task, step })));
  frames.push({ task: tasks.length - 1, step: 'policy' });
  const repeatRef: CodeRef = { symbol: 'runLoop()', path: loopFile, from: 258, to: 299 };
  let frameIndex = 0;
  function render(index: number) {
    frameIndex = index;
    const { task: taskIndex, step } = frames[index], task = tasks[taskIndex];
    const shared = step === 'cli' || step === 'policy', model = step === 'choose';
    const mapping: Record<Phase, { title: string; direct: string; pi: string; refs: CodeRef[]; message: string }> = {
      request: { title: 'A · 판단 모델 요청', direct: 'requestNextStep(context, tools)', pi: 'streamAssistantResponse()', refs: [sourceRefs.model], message: `${task.title} · 작업 명세와 앞선 결과를 판단 모델에 보냅니다.` },
      choose: { title: '모델의 선택 · 두 방식 동일', direct: `${task.tool}()`, pi: `${task.tool}()`, refs: [sourceRefs.toolCall], message: `판단 모델: “${task.tool} 도구를 요청합니다.”` },
      dispatch: { title: 'B · 검증·실행 연결', direct: 'validateCall() → dispatchTool()', pi: 'prepareToolCall() → executePreparedToolCall()', refs: [sourceRefs.validate, sourceRefs.execute], message: `${task.tool}의 이름과 입력을 확인한 뒤 등록한 실행 함수에 연결합니다.` },
      cli: { title: '두 방식 모두 · 직접 구현', direct: task.runner, pi: task.runner, refs: [sourceRefs.execute], message: `내 연결 코드 → ${task.title}. 프로세스 시작과 출력 해석은 직접 구현합니다.` },
      result: { title: 'C · 결과 메시지 연결', direct: 'appendToolResult() · emitProgress()', pi: 'createToolResultMessage()', refs: [sourceRefs.result], message: `결과 연결 · ${task.result}` },
      repeat: { title: 'D · 다음 요청으로 연결', direct: 'while (...) { request → execute → result }', pi: 'runLoop()', refs: [repeatRef], message: task.next },
      policy: { title: '최종 확인 · 두 방식 모두 직접 구현', direct: 'checkLatestVersion(v2)', pi: 'checkLatestVersion(v2)', refs: [sourceRefs.subscribe], message: 'v2의 리뷰·검사 결과가 모두 있는지 내 코드가 확인합니다. 개발자가 최종 반영을 결정합니다.' },
    };
    const current = mapping[step];
    root.dataset.comparePhase = step; root.dataset.compareTask = String(taskIndex); root.dataset.frame = String(index);
    root.querySelectorAll<HTMLElement>('[data-compare-step]').forEach(node => {
      const active = node.dataset.compareStep === step;
      node.dataset.current = String(active);
      if (active) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current');
    });
    milestones.forEach((item, i) => item.dataset.state = i === taskIndex ? 'current' : i < taskIndex ? 'done' : 'next');
    phase.textContent = `${index + 1} / ${frames.length} · ${current.title}`;
    speech.textContent = current.message;
    direct.label.textContent = model ? '직접 연결 · 모델이 판단' : '직접 연결 · 내가 구현';
    pi.label.textContent = model ? 'Pi 사용 · 모델이 판단' : shared ? 'Pi 사용 · 내가 구현' : 'Pi 제공 · 이 코드를 재사용';
    direct.card.dataset.owner = model ? 'model' : 'app';
    pi.card.dataset.owner = model ? 'model' : shared ? 'app' : 'pi';
    direct.code.textContent = current.direct; pi.code.textContent = current.pi;
    direct.refs.textContent = model ? '다음 작업의 선택은 모델의 몫' : shared ? '이 코드는 Pi를 써도 직접 구현' : '직접 작성할 역할의 예시 이름';
    pi.refs.replaceChildren(el('span', '', model ? 'Pi가 이 응답을 받는 곳' : shared ? 'Pi 연결점' : '실제 Pi 구현'));
    current.refs.forEach(ref => {
      const link = el('a', '', `${ref.path.split('/').at(-1)} · ${ref.from}–${ref.to}`);
      link.href = `${sourceBase}${ref.path}#L${ref.from}-L${ref.to}`; link.title = ref.symbol;
      link.target = '_blank'; link.rel = 'noopener noreferrer'; pi.refs.append(link);
    });
  }
  render(0);
  app.play.addEventListener('click', () => {
    void app.run(async signal => {
      for (let i = 0; i < frames.length; i++) { render(i); await wait(1100, signal); }
    });
  });
  next.addEventListener('click', () => { app.stop(); speech.setAttribute('aria-live', 'polite'); render((frameIndex + 1) % frames.length); });
  app.reset.addEventListener('click', () => { app.stop(); render(0); app.status.textContent = '준비'; });
  const pauseAtSource = (event: Event) => { if ((event.target as Element).closest('a')) app.stop(); };
  root.addEventListener('focusin', pauseAtSource); root.addEventListener('click', pauseAtSource);
}

document.querySelectorAll<HTMLElement>('[data-pi-lab]').forEach(root => {
  if (root.dataset.piLab === 'comparison' && !root.dataset.ready) { initComparison(root); return; }
  if (root.dataset.ready || !root.querySelector('[data-lab-mount]')) return;
  if (root.dataset.piLab === 'loop') initReview(root);
  if (root.dataset.piLab === 'context') initHandoff(root);
  if (root.dataset.piLab === 'tools') initReviewers(root);
});
