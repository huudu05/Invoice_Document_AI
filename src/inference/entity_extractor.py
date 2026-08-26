from typing import Dict, List


class EntityExtractor:
    """
    Convert token-level LayoutLMv3 predictions
    into structured invoice entities.

    Supported labels:
        O
        S-ENTITY
        B-ENTITY
        I-ENTITY
        E-ENTITY
    """

    def __init__(self, id2label: Dict[int, str]):

        self.id2label = {
            int(key): value
            for key, value in id2label.items()
        }

        self.entity_types = {
            "company",
            "date",
            "address",
            "total"
        }

    def extract(
        self,
        words: List[str],
        labels: List[int]
    ) -> Dict[str, str]:

        entities = {
            "company": [],
            "date": [],
            "address": [],
            "total": []
        }

        current_entity = None
        current_words = []

        for word, label_id in zip(words, labels):

            label = self.id2label.get(
                label_id,
                "O"
            )

            label = label.upper()

            if label == "O":

                self._flush_entity(
                    entities,
                    current_entity,
                    current_words
                )

                current_entity = None
                current_words = []

                continue

            prefix, entity_name = self._parse_label(
                label
            )

            # Unknown entity
            if entity_name not in self.entity_types:

                self._flush_entity(
                    entities,
                    current_entity,
                    current_words
                )

                current_entity = None
                current_words = []

                continue

            if prefix == "S":

                self._flush_entity(
                    entities,
                    current_entity,
                    current_words
                )

                entities[entity_name].append(
                    word
                )

                current_entity = None
                current_words = []

                continue

            if prefix == "B":

                self._flush_entity(
                    entities,
                    current_entity,
                    current_words
                )

                current_entity = entity_name
                current_words = [word]

                continue

            if prefix == "I":

                if current_entity == entity_name:

                    current_words.append(word)

                else:

                    # Model produced I-* without B-*
                    # Recover gracefully.

                    self._flush_entity(
                        entities,
                        current_entity,
                        current_words
                    )

                    current_entity = entity_name
                    current_words = [word]

                continue

            if prefix == "E":

                if current_entity == entity_name:

                    current_words.append(word)

                else:

                    current_entity = entity_name
                    current_words = [word]

                self._flush_entity(
                    entities,
                    current_entity,
                    current_words
                )

                current_entity = None
                current_words = []

                continue



        self._flush_entity(
            entities,
            current_entity,
            current_words
        )

        return {
            key: " ".join(value).strip()
            for key, value in entities.items()
        }

    @staticmethod
    def _parse_label(label: str):

        if "-" not in label:
            return "S", label.lower()

        prefix, entity_name = label.split(
            "-",
            1
        )

        return (
            prefix.upper(),
            entity_name.lower()
        )

    @staticmethod
    def _flush_entity(
        entities: Dict[str, List[str]],
        entity_name: str,
        words: List[str]
    ):

        if (
            entity_name is not None
            and entity_name in entities
            and words
        ):

            entities[entity_name].extend(
                words
            )