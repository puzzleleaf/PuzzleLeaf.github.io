import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const wiki = args[0] === '--wiki';
if (wiki) args.shift();
const slug = args[0];
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error(`사용법: npm run ${wiki ? 'new-wiki' : 'new-post'} -- my-topic (영문 소문자, 숫자, 하이픈)`);
  process.exit(1);
}
const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
const directory = resolve(`src/content/${wiki ? 'wiki' : 'posts'}`);
await mkdir(directory, { recursive: true });
const path = resolve(directory, `${slug}.md`);
try {
  const body = wiki
    ? '## 핵심 설명\n\n개념이나 사용법을 간단히 정리합니다.\n\n## 사용 조건\n\n확인한 버전과 적용 조건을 적습니다.\n\n## 예시\n\n다시 사용할 코드, 명령어, 설정과 예상 결과를 적습니다.\n\n## 참고 자료\n\n원문 링크와 관련 자료를 남깁니다.\n'
    : '## 개요\n\n대상이 무엇인지, 어떤 문제를 해결하는지, 이 글에서 답할 질문을 설명합니다.\n\n## 분석 기준과 범위\n\n선택한 기준, 확인 날짜·환경·버전·커밋과 분석에서 제외한 범위를 적습니다.\n\n## 동작과 소스 근거\n\n작은 입력을 따라 처리 흐름을 설명하고 핵심 코드의 고정 링크를 연결합니다. 관찰과 해석을 구분합니다.\n\n## 직접 확인해 보기\n\n순서·상태 변화가 핵심일 때 조작 가능한 설명을 넣습니다. 실제 실행과 설명용 시뮬레이션을 구분하고 정적 설명도 남깁니다. 필요하지 않으면 이 절은 생략합니다.\n\n## 설계 판단과 적용\n\n선택의 장점과 비용, 적합한 조건을 설명합니다. 적용 제안은 작성자의 제안이라고 밝힙니다.\n\n## 검증 결과와 한계\n\n실제로 확인한 결과와 확인하지 못한 범위를 구분합니다.\n\n## 출처\n\n원본 저장소, 핵심 소스와 공식 문서를 글 맨 아래에 모읍니다. 작성 원칙: /wiki/writing-principles/\n';
  await writeFile(path, `---\ntitle: "${wiki ? '문서' : '글'} 제목"\ndescription: "문서의 주제와 핵심 내용을 적어주세요."\ndate: ${date}\nsummary: []\ncategory: ${wiki ? '미분류' : '개발 노트'}\ntags: []\n${wiki ? 'related: []\norder: 0\n' : ''}draft: true\n---\n\n${body}`, { flag: 'wx' });
  console.log(`초안 생성: ${path}\n공개하려면 draft를 false로 바꾸세요.`);
} catch (error) {
  if (error.code === 'EEXIST') { console.error('같은 이름의 글이 이미 있습니다.'); process.exit(1); }
  throw error;
}
