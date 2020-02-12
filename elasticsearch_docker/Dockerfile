FROM docker.elastic.co/elasticsearch/elasticsearch:6.7.1

RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install https://artifacts.elastic.co/downloads/elasticsearch-plugins/analysis-icu/analysis-icu-6.7.1.zip

COPY --chown=elasticsearch:elasticsearch utils/wait-for-it.sh /utils/wait-for-it.sh
COPY --chown=elasticsearch:elasticsearch config/add_index_template.sh /usr/local/bin/elasticsearch/
COPY --chown=elasticsearch:elasticsearch config/docker-entrypoint.sh /usr/local/bin/elasticsearch/
COPY --chown=elasticsearch:elasticsearch templates /usr/local/bin/elasticsearch/templates
COPY --chown=elasticsearch:elasticsearch config/analysis /usr/share/elasticsearch/config/analysis
COPY elasticsearch-healthcheck /usr/local/bin/


USER root
RUN chmod +x /utils/wait-for-it.sh /usr/local/bin/elasticsearch/add_index_template.sh /usr/local/bin/elasticsearch/docker-entrypoint.sh /usr/local/bin/elasticsearch-healthcheck

USER elasticsearch
ENTRYPOINT ["/usr/local/bin/elasticsearch/docker-entrypoint.sh"]
CMD ["/usr/share/elasticsearch/bin/elasticsearch"]

HEALTHCHECK --interval=1m --timeout=1s \
    CMD ["elasticsearch-healthcheck"]