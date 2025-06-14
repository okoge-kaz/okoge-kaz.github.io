---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* M.S. in Computer Science, Institute of Science Tokyo 2026 (expected)
* B.S. in Computer Science, Tokyo Institute of Technology, 2024

Work experience
======
* Jun 2025 - Present: Research Engineer Intern
  * Preferred Networks, Inc., Tokyo, Japan
  * Duties included: Research and development of LLM pre-training
  * Supervisor: Hiroaki Mikami, Shuji Suzuki


* Feb 2023 - Present: Research Intern
  * Turing.Inc, Tokyo, Japan
  * Duties included: Developed LLM and VLM training libraries, orchestrated cluster deployments on Google Cloud, managed internal cluster environments
  * Supervisor: Yu Yamaguchi


* Oct 2023 - Present: Research Intern
  * National Institute of Advanced Industrial Science and Technology (AIST), Tokyo, Japan
  * Duties included: Contributed to the Swallow Project, managed pre-training and post-training libraries, created synthetic data
  * Supervisor: Hiroya Takamura


* Apr 2024 - May 2025: Research Intern
  * SB Intuitions, Tokyo, Japan
  * Duties included: Maintained Megatron-LM fork, created datasets for Japanese LLM, researched FP8 training techniques
  * Supervisor: Sho Takase, Toshiaki Hishinuma


* Apr 2024 - Dec 2024: Research Intern
  * Sakana AI, Tokyo, Japan
  * Duties included: Built 35-node training environment with H100 GPUs, adopted Slurm and GCS, introduced Environment Modules
  * Supervisor: Takuya Akiba


* Oct 2023 - Feb 2024: Researcher
  * Kotoba Technologies, Inc., Tokyo, Japan
  * Duties included: Developed custom LLM finetuning library, created Mamba training library, led experiments on Mamba training
  * Supervisor: Jungo Kasai, Noriyuki Kojima


* Aug 2023 - Sep 2023: Summer Intern
  * Preferred Networks, Inc., Tokyo, Japan
  * Duties included: Developed prototype web app with FastAPI and Next.js, deployed on AWS ECS
  * Supervisor: Yuichi Inagaki, Yoichi Yamakawa


* May 2022 - Nov 2022: Backend Developer
  * pixiv Inc., Tokyo, Japan
  * Duties included: Led backend development with Ruby on Rails, modernized legacy APIs, enhanced admin site


* Sep 2022: Cloud Infrastructure Engineer
  * Money Forward, Inc., Tokyo, Japan
  * Duties included: Managed AWS EKS infrastructure, consolidated resources, modified Terraform files


* Aug 2022 - Sep 2022: Software Engineer Intern
  * Cookpad Japan, Kanagawa, Japan
  * Duties included: Redeveloped shopping site with TypeScript and Next.js, optimized performance


Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
