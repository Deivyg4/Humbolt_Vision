document.addEventListener("DOMContentLoaded", () => {
  const chatContainer = document.getElementById("chatContainer")
  const messageForm = document.getElementById("messageForm")
  const messageInput = document.getElementById("messageInput")
  const fileInput = document.getElementById("fileInput")
  const uploadBtn = document.getElementById("uploadBtn")
  const imagePreview = document.getElementById("imagePreview")
  const previewImg = document.getElementById("previewImg")
  const removeImageBtn = document.getElementById("removeImageBtn")

  let selectedFile = null

  // Initialize with welcome message
  scrollToBottom()

  // Event listeners
  messageForm.addEventListener("submit", handleSubmit)
  uploadBtn.addEventListener("click", () => fileInput.click())
  fileInput.addEventListener("change", handleFileSelect)
  removeImageBtn.addEventListener("click", removeSelectedImage)

  // Functions
  function handleSubmit(e) {
    e.preventDefault()

    const message = messageInput.value.trim()

    if (!message && !selectedFile) return

    if (selectedFile) {
      // Send message with image
      addMessage("user", message || "Analiza esta imagen", selectedFile)

      const formData = new FormData()
      formData.append("file", selectedFile)

      // Mostrar mensaje de carga
      const loadingId = showLoading()

      fetch("/analyze", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Ocultar mensaje de carga
          hideLoading(loadingId)

          if (data.error) {
            addMessage("bot", `Error: ${data.error}`)
          } else {
            addMessage("bot", data.message)

            // Si hay detalles, mostrarlos
            if (data.details && data.details.length > 0) {
              let detailsMessage = "Otros posibles resultados:\n"
              data.details.slice(1, 5).forEach((item, index) => {
                detailsMessage += `${index + 2}. ${item.label.replace("_", " ").title()} (${(item.probability * 100).toFixed(2)}%)\n`
              })

              setTimeout(() => {
                addMessage("bot", detailsMessage)
              }, 500)
            }
          }
          scrollToBottom()
        })
        .catch((error) => {
          // Ocultar mensaje de carga
          hideLoading(loadingId)

          console.error("Error:", error)
          addMessage("bot", "Lo siento, hubo un error al procesar tu imagen.")
          scrollToBottom()
        })

      removeSelectedImage()
    } else {
      // Send text message for weather recommendation
      addMessage("user", message)

      fetch("/get_recommendation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ weather_info: message }),
      })
        .then((response) => response.json())
        .then((data) => {
          addMessage("bot", data.recommendation)
          scrollToBottom()
        })
        .catch((error) => {
          console.error("Error:", error)
          addMessage("bot", "Lo siento, hubo un error al procesar tu solicitud.")
          scrollToBottom()
        })
    }

    messageInput.value = ""
  }

  function handleFileSelect(e) {
    const file = e.target.files[0]
    if (file) {
      selectedFile = file
      const reader = new FileReader()
      reader.onload = (e) => {
        previewImg.src = e.target.result
        imagePreview.style.display = "flex"
      }
      reader.readAsDataURL(file)
    }
  }

  function removeSelectedImage() {
    selectedFile = null
    previewImg.src = ""
    imagePreview.style.display = "none"
    fileInput.value = ""
  }

  function showLoading() {
    const loadingId = "loading-" + Date.now()
    const loadingDiv = document.createElement("div")
    loadingDiv.className = "message bot"
    loadingDiv.id = loadingId

    loadingDiv.innerHTML = `
            <div class="message-content">
                <p>Analizando imagen con TensorFlow y PyTorch...</p>
                <div class="loading-indicator">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <div class="message-info">
                    <i class="fas fa-brain"></i>
                    Humbolt Vision
                </div>
            </div>
        `

    chatContainer.appendChild(loadingDiv)
    scrollToBottom()

    return loadingId
  }

  function hideLoading(loadingId) {
    const loadingElement = document.getElementById(loadingId)
    if (loadingElement) {
      loadingElement.remove()
    }
  }

  function addMessage(type, content, image = null) {
    const messageDiv = document.createElement("div")
    messageDiv.className = `message ${type}`

    let messageHTML = `<div class="message-content">`

    if (image && type === "user") {
      const reader = new FileReader()
      reader.onload = (e) => {
        const imgContainer = messageDiv.querySelector(".message-image")
        if (imgContainer) {
          const img = document.createElement("img")
          img.src = e.target.result
          img.alt = "Imagen subida"
          imgContainer.appendChild(img)
        }
      }
      reader.readAsDataURL(image)

      messageHTML += `<div class="message-image"></div>`
    }

    messageHTML += `<p>${content}</p>`

    if (type === "bot") {
      messageHTML += `
                <div class="message-info">
                    <i class="fas fa-brain"></i>
                    Humbolt Vision
                </div>
            `
    }

    messageHTML += `</div>`
    messageDiv.innerHTML = messageHTML

    chatContainer.appendChild(messageDiv)
    scrollToBottom()
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight
  }
})

