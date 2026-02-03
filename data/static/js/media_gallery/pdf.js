/////////////  V2 - Simplified (no iframe modal) \\\\\\\\\\\\

// Global variables for PDFs
let allPdfs = [];
let currentPdfPage = 1;

/**
 * Opens a PDF in a new browser tab
 * @param {string} path - Path of the PDF file
 */
function openPdf(path) {
    const baseUrl = window.cdnFilesUrl || window.location.origin;
    const url = new URL(path, baseUrl);
    if (window.pdfToken) {
        url.searchParams.append('token', window.pdfToken);
    }
    window.open(url.toString(), '_blank');
}

/**
 * Downloads a PDF file
 * @param {string} path - Path of the PDF file
 */
function downloadPdf(path) {
    const baseUrl = window.cdnFilesUrl || window.location.origin;
    const url = new URL(path, baseUrl);
    if (window.pdfToken) {
        url.searchParams.append('token', window.pdfToken);
        url.searchParams.append('download', 'true');
    }
    window.open(url.toString(), '_blank');
}

/**
 * Deletes a single PDF from the server
 * @param {string} encodedPath - URL-encoded path of the PDF to delete
 */
function deletePdf(encodedPath) {
    const path = decodeURIComponent(encodedPath);

    if (confirm('Are you sure you want to delete this PDF?')) {
        const payload = {
            pdf_path: path.startsWith('data/') ? path : `data${path}`
        };

        fetch('/delete-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        })
        .then(async response => {
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            return response.json();
        })
        .then(data => {
            NotificationModal.info('PDF Deleted', data.message);

            // Remove PDF from array and re-render
            allPdfs = allPdfs.filter(pdf => pdf.path !== path && pdf.nginx_path !== path);

            const totalPages = Math.ceil(allPdfs.length / ITEMS_PER_PAGE);
            if (currentPdfPage > totalPages && totalPages > 0) {
                currentPdfPage = totalPages;
            }
            renderPdfPage(currentPdfPage);
        })
        .catch((error) => {
            console.error('Error deleting PDF:', error);
            NotificationModal.error('Delete Error', `Error deleting PDF: ${error.message}`);
        });
    }
}

/**
 * Deletes multiple selected PDFs
 */
function deleteSelectedPdfs() {
    const selectedPdfs = document.querySelectorAll('.pdf-checkbox:checked');
    if (selectedPdfs.length === 0) {
        NotificationModal.warning('Selection Required', 'Please select at least one PDF to delete');
        return;
    }

    if (confirm(`Are you sure you want to delete ${selectedPdfs.length} selected PDFs?`)) {
        const pdfPaths = Array.from(selectedPdfs).map(checkbox => decodeURIComponent(checkbox.dataset.path));

        fetch('/delete-pdfs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pdf_paths: pdfPaths })
        })
        .then(async response => {
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            return response.json();
        })
        .then(data => {
            NotificationModal.info('PDFs Deleted', data.message);
            location.reload();
        })
        .catch(error => {
            console.error('Error deleting PDFs:', error);
            NotificationModal.error('Delete Error', 'An error occurred while deleting the PDFs');
        });
    }
}

/**
 * Loads PDFs from the server
 */
function loadPDFs() {
    fetch('/get-pdfs')
        .then(response => response.json())
        .then(data => {
            allPdfs = data.pdfs;
            window.pdfToken = data.pdf_token;
            renderPdfPage(1);
        })
        .catch(error => {
            console.error('Error loading PDFs:', error);
            NotificationModal.error('Load Error', 'Error loading PDFs');
        });
}

/**
 * Renders a page of PDFs
 * @param {number} page - Page number to render
 */
function renderPdfPage(page) {
    currentPdfPage = page;
    const pageData = paginationUtils.getCurrentPageData(allPdfs, page, ITEMS_PER_PAGE);
    const container = document.getElementById('pdfContainer');

    container.innerHTML = pageData.map(pdf => `
        <div class="pdf-container">
            <div class="pdf-wrapper">
                <input type="checkbox" class="pdf-checkbox" data-path="${encodeURIComponent(pdf.path)}" title="Select for bulk delete">
                <div class="pdf-icon" onclick="openPdf('${pdf.nginx_path}')" title="Click to open PDF">
                    <i class="fas fa-file-pdf"></i>
                </div>
            </div>
            <div class="pdf-info">
                <div class="pdf-name" title="${pdf.name}">${pdf.name}</div>
            </div>
            <div class="pdf-controls">
                <button onclick="downloadPdf('${pdf.nginx_path}')" class="btn btn-sm btn-primary" title="Download PDF">
                    <i class="fas fa-download"></i> Download
                </button>
                <button onclick="deletePdf('${encodeURIComponent(pdf.path)}')" class="btn btn-sm btn-danger" title="Delete PDF">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        </div>
    `).join('');

    // Update pagination controls
    const paginationElement = document.getElementById('pagination-pdfs');
    paginationElement.innerHTML = paginationUtils.createPaginationControls(
        allPdfs.length,
        page,
        ITEMS_PER_PAGE,
        renderPdfPage
    );

    // Add event listeners to pagination controls
    paginationElement.querySelectorAll('.page-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const newPage = parseInt(e.target.dataset.page);
            if (!isNaN(newPage) && newPage > 0 && newPage <= Math.ceil(allPdfs.length / ITEMS_PER_PAGE)) {
                renderPdfPage(newPage);
            }
        });
    });
}

// Utility functions for PDF handling
const pdfUtils = {
    validatePath: function(path) {
        return path && typeof path === 'string' && path.toLowerCase().endsWith('.pdf');
    },
    getFileName: function(path) {
        return path.split('/').pop().split('\\').pop();
    }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const pdfTab = document.getElementById('pdf-tab');
    pdfTab.addEventListener('shown.bs.tab', function (e) {
        if (!pdfTab.dataset.loaded) {
            loadPDFs();
            pdfTab.dataset.loaded = true;
        }
    });

    const deleteSelectedPdfsButton = document.getElementById('deleteSelectedPdfs');
    if (deleteSelectedPdfsButton) {
        deleteSelectedPdfsButton.addEventListener('click', deleteSelectedPdfs);
    }
});

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        openPdf,
        downloadPdf,
        deletePdf,
        pdfUtils
    };
}
