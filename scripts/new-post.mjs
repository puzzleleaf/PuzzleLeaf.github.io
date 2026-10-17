import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const slug = process.argv[2];
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error('사용법: npm run new-post -- my-post-title (영문 소문자, 숫자, 하이픈)');
  process.exit(1);
}
const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
const directory = resolve('src/content/posts');
await mkdir(directory, { recursive: true });
const path = resolve(directory, `${slug}.md`);
try {
  await writeFile(path, `---\ntitle: "글 제목"\ndescription: "이 글에서 다루는 문제와 결론을 한두 문장으로 적어주세요."\ndate: ${date}\ncategory: 개발 노트\ntags: []\ndraft: true\n---\n\n## 핵심 요약\n\n문제와 결론을 짧게 정리합니다.\n\n## 환경과 문제\n\n사용한 버전, 발생 조건, 기대 동작을 기록합니다.\n\n## 해결 과정\n\n재현 가능한 예시와 선택한 해결 방법을 설명합니다.\n\n## 검증 결과와 한계\n\n실제로 확인한 결과와 확인하지 못한 범위를 구분합니다.\n\n## 참고 자료\n\n공식 문서와 참고한 원문을 링크합니다.\n`, { flag: 'wx' });
  console.log(`초안 생성: ${path}\n공개하려면 draft를 false로 바꾸세요.`);
} catch (error) {
  if (error.code === 'EEXIST') { console.error('같은 이름의 글이 이미 있습니다.'); process.exit(1); }
  throw error;
}
