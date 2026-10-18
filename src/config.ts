export const site = {
  title: 'puzzleleaf의 기술블로그',
  author: 'puzzleleaf',
  description: '개발 기록과 개인 지식을 정리하는 puzzleleaf의 블로그와 위키.',
};

// giscus 설정값은 공개 식별자이며 비밀 키가 아닙니다.
// 저장소 Discussions의 Announcements 카테고리를 사용합니다.
export const comments = {
  repo: 'puzzleleaf/PuzzleLeaf.github.io',
  repoId: 'MDEwOlJlcG9zaXRvcnkxMDA2ODUzNzM=',
  category: 'Announcements',
  categoryId: 'DIC_kwDOBgBWPc4DIw16',
};

export const href = (path = '') => `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
export const formatDate = (date: Date) => new Intl.DateTimeFormat('ko-KR', {
  year: 'numeric', month: '2-digit', day: '2-digit', timeZone: 'Asia/Seoul',
}).format(date).replace(/\.$/, '');
