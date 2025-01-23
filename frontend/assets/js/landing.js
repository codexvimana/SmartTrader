window.addEventListener("DOMContentLoaded", () => {
  const initButton = document.querySelector(".init-button");
  const statusMessages = document.querySelectorAll(".status-message");
  const systemStatus = document.querySelector(".system-status");

  if (localStorage.getItem("systemInitialized") === "true") {
    window.location.href = "console.html";
    return;
  }

  const initializationSteps = [
    "SYSTEM INITIALIZING...",
    "SYSTEM READY FOR DEPLOYMENT",
  ];

  initButton.addEventListener("click", async () => {
    initButton.disabled = true;
    initButton.style.opacity = "0.5";

    await runInitializationSequence();
    transitionToConsole();
  });

  async function runInitializationSequence() {
    systemStatus.style.opacity = "1";

    for (let step of initializationSteps) {
      systemStatus.textContent = step;
      await new Promise((resolve) => setTimeout(resolve, 600));
    }

    await sequentialFade(statusMessages);
  }

  async function sequentialFade(elements) {
    for (let element of elements) {
      element.style.opacity = "1";
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }

  function transitionToConsole() {
    document.body.classList.add("doors-open");
    localStorage.setItem("systemInitialized", "true");

    setTimeout(() => {
      window.location.href = "console.html";
    }, 2000);
  }
});
