// JavaScript for toggling sidebar
document.getElementById('hamburgerMenu').addEventListener('click', function () {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
});

function loadImage(event) {
    const image = document.getElementById('uploadedImage');
    const imageName = document.getElementById('imageName');
    const uploadNote = document.querySelector('.image-container p');


    // Display image name below the upload button
    imageName.textContent = 'File name: ' + event.target.files[0].name;

    // Display the image in the container
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.display = 'block';

    // Hide the note about max size upload
    if (uploadNote) {
        uploadNote.style.display = 'none';
    }
}

function generateCaption() {
    const loader = document.getElementById('loader');
    const localCaptionDisplay = document.getElementById('localCaption');
    const groqCaptionDisplay = document.getElementById('groqCaption');
    const instagramContentDisplay = document.getElementById('instagramContent');
    const fileInput = document.getElementById('imageUpload');
    
    localCaptionDisplay.textContent = '';
    groqCaptionDisplay.textContent = '';
    instagramContentDisplay.textContent = '';


    // Check if a file is selected in the file input
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please upload an image first!");
        return; // Stop execution if no file is selected
    }

    // Display loader while generating caption
    loader.style.display = 'block';

    // Prepare the form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Send the image to the server via a POST request
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Failed to generate caption.");
        }
        return response.json();
    })
    .then(data => {
        // Hide loader once data is received
        loader.style.display = 'none';

        // Format the captions and Instagram content
        const formattedContent = formatContent(data.groq_caption, data.instagram_content);

        // Display the results
        localCaptionDisplay.textContent = ''; // Clear previous content
        groqCaptionDisplay.innerHTML = formattedContent; // Display new content

    })
    .catch(error => {
        loader.style.display = 'none';
        console.error("Error:", error);
        alert("An error occurred. Please try again later.");
    });
}

/**
 * Format captions and Instagram content.
 */
function formatContent(groqCaption, instagramContent) {
    const groqSections = parseSections(groqCaption);
    const instaSections = parseSections(instagramContent);

    // Combine Content for Insta (groqCaption + instagramContent)
    const instaContents = [
        groqSections.caption,
        groqSections.firstComment,
        instaSections.firstComment
    ].filter(Boolean).map((content, index) => `${index + 1}. ${content}`).join('<br>');

    // Combine hashtags
    const combinedHashtags = [...(groqSections.hashtags || []), ...(instaSections.hashtags || [])].join(' ');

    return `
    <span style="font-size: 28px; font-weight: bold;">📸 Discover Engaging Content for your Post Perfectly! 🎯 </span><br>
    <hr>
    ${instaContents}<br><br>
    <span style="font-size: 28px; font-weight: bold;">✨ Level Up Your Engagement With Optimized Hashtags 🔥</span><br>
    <hr>
    ${combinedHashtags}
     `;
}

/**
 * Parse sections from the API response text.
 */
function parseSections(text) {
    const sections = text.split('###').map(section => section.trim()).filter(Boolean);
    const result = {};

    sections.forEach(section => {
        const [title, ...content] = section.split('\n').filter(Boolean);
        const key = title.toLowerCase().includes('caption')
            ? 'caption'
            : title.toLowerCase().includes('hashtags')
            ? 'hashtags'
            : 'firstComment';

        result[key] = key === 'hashtags' ? content.join(' ').split(' ') : content.join(' ');
    });

    return result;
}