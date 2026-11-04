
function extractComments() {
  const elements = document.querySelectorAll(
    "ytd-comment-thread-renderer #content-text"
  );

  const comments = Array.from(elements)
    .map(el => el.innerText.trim())
    .filter(text => text.length > 0);

  return [...new Set(comments)];
}

chrome.runtime.onMessage.addListener(
  (request, sender, sendResponse) => {
    if (request.action === "GET_COMMENTS") {
      sendResponse({
        comments: extractComments()
      });
    }
  }
);
