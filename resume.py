%-------------------------
% Resume in Latex
% Author : Abey George
% Based off of: https://github.com/sb2nov/resume
% License : MIT
%------------------------

\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage[english]{babel}
\usepackage{tabularx}
\usepackage{fontawesome5}
\usepackage{multicol}
\usepackage{graphicx}
\setlength{\multicolsep}{-3.0pt}
\setlength{\columnsep}{-1pt}
\input{glyphtounicode}

\RequirePackage{tikz}
\RequirePackage{xcolor}
\RequirePackage{fontawesome}
\usepackage{tikz}
\usetikzlibrary{svg.path}


\definecolor{cvblue}{HTML}{0E5484}
\definecolor{black}{HTML}{130810}
\definecolor{darkcolor}{HTML}{0F4539}
\definecolor{cvgreen}{HTML}{3BD80D}
\definecolor{taggreen}{HTML}{00E278}
\definecolor{SlateGrey}{HTML}{2E2E2E}
\definecolor{LightGrey}{HTML}{666666}
\colorlet{name}{black}
\colorlet{tagline}{darkcolor}
\colorlet{heading}{darkcolor}
\colorlet{headingrule}{cvblue}
\colorlet{accent}{darkcolor}
\colorlet{emphasis}{SlateGrey}
\colorlet{body}{LightGrey}



%----------FONT OPTIONS----------
% sans-serif
% \usepackage[sfdefault]{FiraSans}
% \usepackage[sfdefault]{roboto}
% \usepackage[sfdefault]{noto-sans}
% \usepackage[default]{sourcesanspro}

% serif
% \usepackage{CormorantGaramond}
% \usepackage{charter}


% \pagestyle{fancy}
% \fancyhf{}  % clear all header and footer fields
% \fancyfoot{}
% \renewcommand{\headrulewidth}{0pt}
% \renewcommand{\footrulewidth}{0pt}

% Adjust margins
\addtolength{\oddsidemargin}{-0.6in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1.19in}
\addtolength{\topmargin}{-.7in}
\addtolength{\textheight}{1.4in}

\urlstyle{same}

\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

% Sections formatting
\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large\bfseries
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

% Ensure that generate pdf is machine readable/ATS parsable
\pdfgentounicode=1

%-------------------------
% Custom commands
\newcommand{\resumeItem}[1]{
  \item\small{
    {#1 \vspace{-2pt}}
  }
}

\newcommand{\classesList}[4]{
    \item\small{
        {#1 #2 #3 #4 \vspace{-2pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{1.0\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{\large#1} & \textbf{\small #2} \\
      \textit{\large#3} & \textit{\small #4} \\
      
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubSubheading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \textit{\small#1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-7pt}
}


\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{1.001\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & \textbf{\small #2}\\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}

\renewcommand\labelitemi{$\vcenter{\hbox{\tiny$\bullet$}}$}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.0in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}


\newcommand\sbullet[1][.5]{\mathbin{\vcenter{\hbox{\scalebox{#1}{$\bullet$}}}}}

%-------------------------------------------
%%%%%%  RESUME STARTS HERE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%


\begin{document}

%----------HEADING----------


\begin{center}
    {\Huge \scshape Krishna Jaiswal} \\ 
   \underline{} ~} \href{krishnajaiswal365@gmail.com}
   \underline{krishnajaiswal365@gmail.com}} ~ 

   
    %\href{https://codeforces.com/profile/yourid}{\raisebox{-0.2\height}\faPoll\ \underline{yourid}}
    \vspace{-8pt}
\end{center}








%-----------EDUCATION-----------
\section{EDUCATION}
\resumeSubHeadingListStart

  \resumeSubheading
    {Axis Institute Of Technology And Management}{2022 -- 2025}
    {B.Tech in Data Science \hspace{0.5cm} \textbf{6.89 GPA}}{Kanpur, India}

  \resumeSubheading
    {Mahamaya Polytechnic Of Information Technology}{2019 -- 2022}
    {Diploma in Computer Science Engineering \hspace{0.5cm} \textbf{76\%}}{Siddharth Nagar, India}

  \resumeSubheading
    {MPIC Inter College}{2015 -- 2017}
    {High School \hspace{0.5cm} \textbf{68\%}}{Rudhauli, India}

\resumeSubHeadingListEnd




%------RELEVANT COURSEWORK / SKILLS-------
\section{COURSEWORK / SKILLS}
\begin{multicols}{4}
  \begin{itemize}[itemsep=-2pt, parsep=5pt]
    \item Python 
    \item Machine Learning 
    \item Exploratory Data Analysis (EDA)
    \item RAG 
    \item DAX 
    \item MySQL 
    \item Power BI 
    \item Microsoft Azure 
    \item Data Warehousing 
    \item Excel 
    \item NLP 
    \item Power Query 
    \item ETL 
    \item Web Scraping 
    \item MCP
  \end{itemize}
\end{multicols}
\vspace*{2.0\multicolsep}


%-----------EXPERIENCE-----------
%\section{INTERNSHIP}
%\resumeSubHeadingListStart

 % \resumeSubheading
  %  {Education Culture Pvt. Ltd. Ludhiana\href{https://educationculture.com}%{\raisebox{-0.1\height}\faExternalLink}}{ Feb 2025 -- may 2025}
    %{\underline{“Python Developer Cum Technical Writer”}}{Hybrid }
    %\resumeItemListStart
     % \resumeItem{\normalsize{Developed and deployed over 20 end-to-end data-driven solutions, integrating Power BI dashboards, advanced Excel models, and NLP pipelines for business use cases.}}
      %\resumeItem{\normalsize{Built machine learning and deep learning models using \textbf{CNN, RNN, and BERT}  implemented with \textbf{TensorFlow and Hugging Face Transformers } to solve text classification and prediction problems.}.}}
      %\resumeItem{\normalsize{Created and published \textbf{dynamic dashboards }in Power BI and Tableau, enabling real-time decision-making for non-technical stakeholders.}}
      
      %\resumeItem{\normalsize{Collaborated with cross-functional teams including analysts, developers, and domain experts to translate business requirements into data science solutions.

}}
    %\resumeItemListEnd
    %-----------INTERNSHIP-----------
\section{Experience}

\resumeSubHeadingListStart

\resumeSubheading
  {AIonOS, Gurgaon, Haryana  \href{\faExternalLink}}{Aug 2025 -- Present}


\resumeSubHeadingListEnd

\resumeSubHeadingListStart

\resumeSubheading
  {Education Culture Pvt. Ltd., Ludhiana \href{https://educationculture.com}{\faExternalLink}}{Feb 2025 -- July 2025}
  {\underline{Python Developer Cum Technical Writer}}{Hybrid}
\resumeItemListStart
  \resumeItem{Designed and deployed 10+ end-to-end analytics solutions using Python, Power BI, and Excel, improving client reporting speed by 40\%.}
  \resumeItem{Developed advanced NLP pipelines using \textbf{BERT}, \textbf{CNN}, and \textbf{RNN} architectures achieving up to 92\% accuracy in text classification tasks.}
 
  \resumeItem{Created and implemented interactive dashboards in  \textbf{Power BI} and \textbf{Tableau}to monitor key sales and customer KPIs across 3 business units, reducing manual reporting time by 30\% .}

  

  \resumeItem{  Created clear, organized documentation for ML models and experiments to assist team collaboration and knowledge sharing.}

  \resumeItem{Conducted A/B testing and statistical analysis on product recommendation models in a project setting to evaluate performance and optimize user relevance.}
  \resumeItem{Used TensorFlow and Hugging Face Transformers to deploy scalable ML models on Flask.}
  \resumeItem{Visualized model performance using confusion matrices and ROC curves for 10K+ sample datasets.}
  \resumeItem{Enhanced precision and recall scores by 15\% through iterative hyperparameter tuning and stratified cross-validation.}
\resumeItemListEnd

\resumeSubHeadingListEnd





%-----------PROJECTS-----------
\section{Projects}
\vspace{-5pt}
\resumeSubHeadingListStart

% -- Pizza Prices Prediction --
\resumeProjectHeading
  {\textbf{Pizza Prices Prediction} 
  \href{https://github.com/krishnajaiswal365/machine_learing_project/blob/main/pizza_new_data.ipynb}{\faExternalLink}
  \hfill \textnormal{Pandas, NumPy, Matplotlib, Seaborn, Sklearn}}{Nov 2024}
\resumeItemListStart
  \resumeItem{Conducted extensive exploratory data analysis on a pizza pricing dataset (1,200+ rows) using univariate and multivariate techniques to identify pricing drivers.}

  \resumeItem{Cleaned and transformed raw data, reducing missing values by 95\% through advanced imputation strategies.}
  \resumeItem{Employed Matplotlib and Seaborn to create insightful visualizations of feature relationships, leading to the identification of five key variables that accounted for 80\% of pizza price variance.}
  \resumeItem{Engineered and fine-tuned regression models including Random Forest, XGBoost, and SVR, achieving a stellar R² score of 0.91, indicating strong model performance.}
  \resumeItem{Applied feature selection techniques to reduce dimensionality by 35\%, resulting in a 12\% boost in model accuracy.}
  \resumeItem{Evaluated and optimized models using cross-validation and residual analysis, improving accuracy by 78\% and reducing RMSE by 0.56\%

}

  \resumeItem{\href{https://github.com/krishnajaiswal365/machine_learing_project/blob/main/pizza_new_data.ipynb}{\textcolor{accent}{Project Link}}}
\resumeItemListEnd
\vspace{-10pt}

% -- Global Superstore Dashboard --
\resumeProjectHeading
  {\textbf{Global Superstore Dashboard}
  \href{https://drive.google.com/file/d/1L4q7RMrbsJH3l8pcSv5XO35zCseTcdYk/view?usp=drive_link}{\faExternalLink}
  \hfill \textnormal{Power BI, DAX, Power Query}}{Aug 2024}
\resumeItemListStart
  \resumeItem{Designed and deployed an interactive Power BI dashboard for a dataset with 10K+ transactions across 13 global regions.}
  \resumeItem{Created 5+ advanced DAX measures in Power BI to analyze profit margins, year-over-year sales trends, and shipping delays across a dataset of 5,000+ transactions.}
  \resumeItem{Merged and cleaned data from 3+ sources using Power Query, reducing manual effort by 50\%.}
  \resumeItem{Scheduled automated data refreshes and report delivery for 5 key stakeholders, enabling real-time insights.

}
  \resumeItem{Elevated analysis efficiency by 25\% by adding interactive KPIs, slicers, and drill-through pages in Power BI, enabling faster insight extraction from multi-level sales data.}
  \resumeItem{\href{https://drive.google.com/file/d/1L4q7RMrbsJH3l8pcSv5XO35zCseTcdYk/view?usp=drive_link}{\textcolor{accent}{Download Report}}}
\resumeItemListEnd
\vspace{-10pt}

% -- Blogging Engineering Hub --
\resumeProjectHeading
  {\textbf{Blogging Engineering Hub}
  \href{ProjectLink.com}{\faExternalLink}
  \hfill \textnormal{Django, HTML/CSS, SQLite}}{Jun 2024}
\resumeItemListStart
  \resumeItem{Constructed a responsive blogging web app with secure login (hashed passwords), CRUD post management, and admin controls; improved user experience across desktop for 5+ user roles.}
  \resumeItem{Built an image upload feature and basic post management system using Django, which Boosted user interaction by 40\% during project testing.}
  \resumeItem{Used Bootstrap and CSS media queries to make a web app fully responsive, reducing layout issues on small screens by 90\% during testing across Chrome DevTools and mobile devices.}
  \resumeItem{Formulated secure authentication flow using Django’s auth system with hashed passwords, session cookies, and CSRF protection, enabling safe login/logout processes and user-specific content access.}
\resumeItemListEnd

\resumeSubHeadingListEnd


%

 %\section{EXPERIENCE}
  %\resumeSubHeadingListStart

    %\resumeSubheading
      %{Eucoders Technologies Pvt Ltd \href{certificate Link}{\raisebox{-0.1\height}\faExternalLink }}{04 2022 -- 06 2023} 
      %{\underline{Python Developer}}{Nishat Ganj, Lucknow,, india}
     % \resumeItemListStart
       % \resumeItem{\normalsize{Developed robust and scalable Python applications, contributing to the full software development lifecycle from design to deployment.}}
       % \resumeItem{\normalsize{Utilized Python frameworks such as Django and Flask to build web applications with front-end systems.}}
        %\resumeItem{\normalsize{Designed and optimized database schemas using MYSQL efficient data storage and retrieval.}}
        %\resumeItem{\normalsize{Collaborated with cross-functional teams to gather requirements,  and deliver high-quality solutions within project deadlines.s.}}



 






  \resumeSubHeadingListStart

%-----------TECHNICAL SKILLS-----------
\section{TECHNICAL SKILLS}
\begin{itemize}[leftmargin=0.15in, label={}]
  \small{\item{
    \textbf{\normalsize{Programming Languages:}} {\normalsize Python, Machine Learning, C, SQL} \\
    \textbf{\normalsize{Data Analytics \& BI Tools:}} {\normalsize Power BI, Microsoft Excel (Advanced), Tableau, Google Colab, Jupyter Notebook} \\
    \textbf{\normalsize{Developer Tools \& IDEs:}} {\normalsize Visual Studio Code, PyCharm, GitHub} \\
    \textbf{\normalsize{Concepts \& Techniques:}} {\normalsize Machine Learning, Deep Learning, Natural Language Processing (NLP), CNN, RNN, Data Cleaning, Feature Engineering, Model Evaluation, ETL, Data Warehousing, DAX, Power Query} \\
  }}
\end{itemize}



 \vspace{-15pt}
%-----------INTERNSHIP-----------















%-----------INVOLVEMENT---------------
%\section{EXTRACURRICULAR}
 %   \resumeSubHeadingListStart
  %      \resumeSubheading{Organization Name \href{Certificate Proof link}{\raisebox{-0.1\height}\faExternalLink } }{MM YYYY -- MM YYYY}{\underline{Role}}{Location}
   %         \resumeItemListStart
    %            \resumeItem{\normalsize{About the role \textbf{and responsibilities carried out.}}}
     %           \resumeItem{\normalsize{Participation Certificate. \href{ParticipationCertificateLink.com}{\raisebox{-0.1\height}\faExternalLink }}}
            \resumeItemListEnd
    \resumeSubHeadingListEnd
 \vspace{-8pt}
 
 %-----------CERTIFICATIONS---------------

 
\section{CERTIFICATIONS}

$\sbullet[.75] \hspace{0.1cm}$ {\href{https://drive.google.com/file/d/1nhkAuAlmM_-Czdm-B1lghTAIAZI6tefd/view?usp=drive_link}{ Data Science - ThinkNEXT Technologies}} \hspace{1.6cm}
$\sbullet[.75] \hspace{0.1cm}$ {\href{https://drive.google.com/file/d/1DlxYKqBno60tmZq4hkrMd9eYBFEsJAkj/view?usp=drive_link}{Python -- ThinkNEXT Technologies}} \hspace{2.59cm}
$\sbullet[.75] \hspace{0.2cm}${\href{https://drive.google.com/file/d/1igLAvbkLt4VrnQwKeN537ULoiOX3p6YL/view?usp=drive_link} {Google Data Analytics -- Coursera}}\\

$\sbullet[.75] \hspace{0.2cm}${\href{certificateLink.com}{{Machine Learning - ThinkNEXT Technologies }} \hspace{1cm}
$\sbullet[.75] \hspace{0.1cm}$ {\href{https://drive.google.com/file/d/1_fTvH16Qv_a1SxF8TtVLqryXALiYKei-/view?usp=drive_link}{SQL}} \hspace{2.6cm}
$\sbullet[.75] \hspace{0.2cm}${\href{https://drive.google.com/file/d/1OOSNS-v_N7gMpyQi_e6RLw8ulgbwkM7e/view?usp=drive_link}{Data Analytics and Visualization -- Accenture}} \\

$\sbullet[.75] \hspace{0.2cm}${\href{https://www.linkedin.com/in/krishna-jaiswal-b6225021b/recent-activity/all/}{\textbf{Data } Analyst -- Skilledge  \href{certificateLink.com}{\raisebox{-0.1\height}\faExternalLink }}}\hspace{1.45cm}
$\sbullet[.75] \hspace{0.2cm}${\href{https://drive.google.com/file/d/1T9zGmasjXdQ8z2Qu9M1YNqXdVZyOpWlJ/view?usp=drive_link}{MongoDB Atlas -- MongoDB }} \hspace{0.5cm}
%$\sbullet[.75] \hspace{0.2cm}${\href{certificateLink.com}{NodeJS with Express \& MongoDB - Udemy}} \\


\end{document}






