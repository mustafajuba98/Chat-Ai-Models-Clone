let currentChatId = null;
let isStreaming = false;

// Model selection
document.getElementById("modelSelect").addEventListener("change", function (e) {
  localStorage.setItem("selectedModel", e.target.value);
});

// Message controls
async function regenerateResponse() {
  if (!currentChatId) return;

  const response = await fetch(`/api/chat/regenerate/${currentChatId}`, {
    method: "POST",
  });
  const data = await response.json();

  if (data.error) {
    appendMessage("error", data.error);
  } else {
    appendMessage("assistant", data.response);
  }
}

async function editLastMessage() {
  const messageDiv = document.querySelector(".user-message:last-child");
  if (!messageDiv) return;

  const content = messageDiv.querySelector(".message-content").textContent;
  const userInput = document.getElementById("userInput");
  userInput.value = content;
  userInput.focus();

  // Remove last user and assistant messages
  messageDiv.remove();
  const lastAssistant = document.querySelector(".assistant-message:last-child");
  if (lastAssistant) lastAssistant.remove();
}

// File handling
document.getElementById("uploadImage").addEventListener("click", function () {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = handleImageUpload;
  input.click();
});

async function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async function (e) {
    const img = document.createElement("img");
    img.src = e.target.result;
    img.className = "uploaded-image";

    appendMessage("user", "", img);
    // Handle image analysis with the selected model
  };
  reader.readAsDataURL(file);
}

// Voice input
const recognition = new (window.SpeechRecognition ||
  window.webkitSpeechRecognition)();
recognition.continuous = false;
recognition.interimResults = false;

document.getElementById("voiceInput").addEventListener("click", function () {
  recognition.start();
});

recognition.onresult = function (event) {
  const transcript = event.results[0][0].transcript;
  document.getElementById("userInput").value = transcript;
};

// ... rest of existing JavaScript code ...
