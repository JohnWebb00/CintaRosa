# DIT826-group1-part2



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.chalmers.se/courses/dit826/2023/group1/dit826-group1-part2.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.chalmers.se/courses/dit826/2023/group1/dit826-group1-part2/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

# CINTA ROSA
Revolutionary breast cancer detection online.


# Introduction
### **Project Overview** 
Our project revolves around the development of a cutting-edge breast cancer detection website. Harnessing the power of Convolutional Neural Networks (CNN), the platform is designed to analyze medical images, specifically those of breast tissue, with the goal of swiftly and accurately identifying potential tumors. This technological innovation has the potential to significantly improve the early detection of breast cancer, offering a valuable tool for individuals alike.

### **Purpose and Goals**
The primary purpose of our project is to contribute to the early detection of breast cancer, a crucial factor in improving patient outcomes. By leveraging advanced algorithms, we aim to automate the analysis of breast tumor characteristics, classifying them as malignant, benign, or normal. Our goals include providing a user-friendly platform that simplifies the detection process, ensuring accessibility for the general public. Through this project, we aspire to make a meaningful impact on the efficiency and accuracy of breast cancer diagnosis, ultimately leading to better prognosis and timely medical interventions.

# Getting Started

### **Installation**
1. Go to https://git.chalmers.se/courses/dit826/2023/group1/dit826-group1
2. Clone the project into a directory of your choosing using either SSH or HTTPS
3. To run the project, open the project on terminal and run the following steps:
```bash
cd projectOne
```
```bash
python manage.py runserver
```
### **System Requirements**

##### Minimum Requirements

- **Operating System:** Windows 10, macOS 10.14, Ubuntu 18.04 LTS, or a compatible OS
- **Processor:** Dual-core processor, 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 20 GB available space
- **Internet Connection:** Required for initial setup and updates

##### Recommended Specifications

- **Operating System:** Windows 10, macOS 10.15, Ubuntu 20.04 LTS, or a compatible OS
- **Processor:** Quad-core processor, 3.0 GHz
- **RAM:** 8 GB or higher
- **Storage:** 50 GB available space (SSD recommended for faster performance)
- **Graphics:** Dedicated graphics card with 2 GB VRAM
- **Internet Connection:** Required for regular updates and access to online features

##### Software Dependencies

- Python 3.8 or higher
- Docker (optional, for containerized deployment)

### **Dependencies**
- **Django and Web Framework:**

    - Django==4.2.7
    - django-environ>=0.7,<1.0
    - django-bootstrap-v5==1.0.11
    - django-bootstrap5
    - djangorestframework-simplejwt
    - djangorestframework
    - django-cors-headers

- **Model Dependencies:**

    - cppimport
    - pybind11
    - seaborn
    - scikit-learn
    - numpy
    - pandas
    - matplotlib
    - eli5
    - pydot
    - tensorflow
    - opencv-python

To ensure smooth execution of the project, follow these steps:

1. **Install Python and pip:**
   Make sure you have Python and pip installed on your system.

2. **Create a Virtual Environment:**
   It's recommended to use a virtual environment to isolate your project dependencies. Run the following commands:

   ```bash
   python -m venv venv
   source venv/bin/activate   
   #On Windows, use `venv\Scripts\activate`

3. **Install Dependencies:**
Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Migrations (Django Only):**
If you're using Django, perform the database migrations:
   ```bash
   python manage.py migrate
   ```

5. Now you're all set! Your environment should be ready to run the project smoothly.
### **For Users**
#### How to use the website (For regular users)

1. Head over to our website
2. On the main page click on LOGIN button if you already have an account, otherwise REGISTER yourself
![homePage](screenshots/home-page.png)
![userLogin](screenshots/user-login.png)
3. On the prediction page, click on the `choose file` to upload your ultrasonic picture of your breast (.png) and the click on `show result` to see the result
![predictionPage](screenshots/prediction-page.png)
4. On the user dashboard page, you can view the upload history of your images, including associated date and corresponding results
![userDashboard](screenshots/user-dashboard.png)
Clicking on the pink box on the left side of the page will navigate you to the prediction page, allowing you to upload additional images

#### How to use the website (For administrators)
On the admin dashboard page, administrators have three options that contribute to model retraining, ultimately enhancing model accuracy
![adminDashboard](admin-dashboard.png)

1. **Upload image(s):** On the left side, you have the option to upload images. Images can be uploaded either in batches or as individual files. If you choose to upload a batch of images, they should be in the form of zip files. For single-image uploads, ensure the image is in the PNG file format

    **NOTE**: The zip file you wish to upload must contain PNG files exclusively. If there are any other image formats inside, the website will only unzip the PNG files and proceed with the upload.

2. **Retrain:** After uploading images, in the middle section of the page, the admin can choose to click on 'Retrain' the model.

3. **Table of models:** You can explore various models and select one based on your preference, also viewing each model's accuracy.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
